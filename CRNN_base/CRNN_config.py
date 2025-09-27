import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import pandas as pd
import numpy as np
import librosa
import wandb
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not available, skip
# Data preprocessing parameters
AUDIO_CONFIG = {
    'sr': 22050,
    'duration': 5,
    'n_fft': 2048,
    'hop_length': 512,
    'n_mels': 96,
    'fmin': 20,
    'fmax': 10000,
    'window': 'hann'
}

# Derived parameters
AUDIO_CONFIG['samples_per_segment'] = AUDIO_CONFIG['sr'] * AUDIO_CONFIG['duration']
AUDIO_CONFIG['exp_vectors_per_segment'] = math.floor(AUDIO_CONFIG['samples_per_segment'] / AUDIO_CONFIG['hop_length'])
AUDIO_CONFIG['exp_input_shape'] = (AUDIO_CONFIG['n_mels'], AUDIO_CONFIG['exp_vectors_per_segment'])

# Dataset paths
DATA_CONFIG = {
    'csv_path': 'dataset/sample_meta.csv',
    'audio_dir': 'dataset/audio/audio',
    'results_dir': f"results/CRNN_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
}

# Wandb configuration
WANDB_CONFIG = {
    'entity': None,  # Use default entity (your username)
    'project': 'drone-detection',
    'tags': ['crnn', 'aircraft-detection'],
    'notes': 'CRNN baseline for aircraft detection'
}

# Model architecture parameters - optimized for aircraft detection
MODEL_CONFIG = {
    'input_channels': 1,
    'n_mels': AUDIO_CONFIG['n_mels'],
    'time_steps': AUDIO_CONFIG['exp_vectors_per_segment'],
    'conv_channels': [16, 32, 64, 128],  # Smaller than current [32, 64, 128, 256]
    'conv_kernel_size': (3, 3),
    'pool_size': (2, 2),
    'dropout_conv': 0.1,  # Reduce from 0.2
    'rnn_hidden_size': 64,  # Reduce from 128
    'rnn_num_layers': 1,  # Reduce from 2
    'rnn_dropout': 0.2,  # Reduced from 0.3
    'attention_heads': 2,  # Reduce from 4
    'attention_dim': 128,  # Reduced from 256
    'fc_hidden_size': 128,  # Reduce from 256
    'dropout_fc': 0.3  # Reduce from 0.4
}

# Training parameters
TRAINING_CONFIG = {
    'batch_size': 80,
    'max_epochs': 50,
    'learning_rate': 5e-4,  # Reduce from 5e-3
    'weight_decay': 1e-5,
    'patience_early_stop': 35,  # CHANGED: from 25 -> 35 (allow more epochs to use full dataset)
    'patience_lr_reduce': 5,    # CHANGED: from 5 -> 3 (reduce LR faster when plateau detected)
    'lr_reduce_factor': 0.5,    # More aggressive reduction from 0.5
    'min_lr': 1e-6,             # Higher than 1e-7
    'min_delta': 0.001,
    'num_workers': 6,
    'pin_memory': True
}

# Task configuration - focused on aircraft detection only
TASK_CONFIG = {
    'task_name': 'train',
    'num_classes': 1,  # Binary classification with sigmoid
    'loss_type': 'binary',
    'class_weights': {0: 1.0, 1: 2.03}  # Adjusted for class imbalance (1270 vs 625)
}


class CBAM(nn.Module):
    """Convolutional Block Attention Module"""
    def __init__(self, channels, reduction=16):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention()
    
    def forward(self, x):
        x = self.channel_attention(x) * x
        x = self.spatial_attention(x) * x
        return x


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_attention=False):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 
                             MODEL_CONFIG['conv_kernel_size'], padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(MODEL_CONFIG['pool_size'])
        self.dropout = nn.Dropout2d(MODEL_CONFIG['dropout_conv'])
        self.attention = CBAM(out_channels) if use_attention else None
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        if self.attention:
            x = self.attention(x)
        x = self.pool(x)
        x = self.dropout(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        batch_size, seq_len, d_model = x.size()
        
        Q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attention_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attention_weights, V)
        
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        output = self.w_o(context)
        
        # Global average pooling over time dimension
        return torch.mean(output, dim=1)


class CRNNModel(pl.LightningModule):
    def __init__(self, task_name, num_classes, loss_type='binary', class_weights=None):
        super(CRNNModel, self).__init__()
        self.task_name = task_name
        self.num_classes = num_classes
        self.loss_type = loss_type
        self.save_hyperparameters()
        
        # CNN frontend
        channels = [MODEL_CONFIG['input_channels']] + MODEL_CONFIG['conv_channels']
        self.conv_blocks = nn.ModuleList()
        
        for i in range(len(MODEL_CONFIG['conv_channels'])):
            use_attention = i >= 2  # Add attention after 3rd and 4th conv blocks
            self.conv_blocks.append(
                ConvBlock(channels[i], channels[i+1], use_attention=use_attention)
            )
        
        # Calculate feature dimensions after conv layers
        self._calculate_conv_output_size()
        
        # Bidirectional GRU
        self.rnn = nn.GRU(
            input_size=self.conv_output_features,
            hidden_size=MODEL_CONFIG['rnn_hidden_size'],
            num_layers=MODEL_CONFIG['rnn_num_layers'],
            batch_first=True,
            dropout=MODEL_CONFIG['rnn_dropout'] if MODEL_CONFIG['rnn_num_layers'] > 1 else 0,
            bidirectional=True
        )
        
        # Multi-head attention pooling
        rnn_output_size = MODEL_CONFIG['rnn_hidden_size'] * 2  # bidirectional
        self.attention = MultiHeadAttention(rnn_output_size, MODEL_CONFIG['attention_heads'])
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(rnn_output_size, MODEL_CONFIG['fc_hidden_size']),
            nn.ReLU(),
            nn.Dropout(MODEL_CONFIG['dropout_fc']),
            nn.Linear(MODEL_CONFIG['fc_hidden_size'], num_classes if loss_type == 'multiclass' else 1)
        )
        
        # Loss function
        if class_weights is not None:
            weights = torch.FloatTensor(list(class_weights.values()))
            if loss_type == 'binary':
                pos_weight = weights[1] / weights[0]
                self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            else:
                self.criterion = nn.CrossEntropyLoss(weight=weights)
        else:
            self.criterion = nn.BCEWithLogitsLoss() if loss_type == 'binary' else nn.CrossEntropyLoss()
    
    def _calculate_conv_output_size(self):
        # Calculate output size after conv layers
        h, w = MODEL_CONFIG['n_mels'], MODEL_CONFIG['time_steps']
        for _ in range(len(MODEL_CONFIG['conv_channels'])):
            h = h // MODEL_CONFIG['pool_size'][0]
            w = w // MODEL_CONFIG['pool_size'][1]
        
        self.conv_output_height = h
        self.conv_output_width = w
        self.conv_output_features = MODEL_CONFIG['conv_channels'][-1] * h
    
    def forward(self, x):
        # CNN frontend
        for conv_block in self.conv_blocks:
            x = conv_block(x)
        
        # Reshape for RNN: (batch, channels*height, width) -> (batch, width, channels*height)
        batch_size = x.size(0)
        x = x.view(batch_size, -1, self.conv_output_width)
        x = x.transpose(1, 2)  # (batch, time_steps, features)
        
        # RNN
        rnn_out, _ = self.rnn(x)
        
        # Attention pooling
        attended = self.attention(rnn_out)
        
        # Classification
        logits = self.classifier(attended)
        
        return logits
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        
        if self.loss_type == 'binary':
            y = y.float().unsqueeze(1)
            loss = self.criterion(logits, y)
            preds = torch.sigmoid(logits) > 0.5
            acc = (preds == y).float().mean()
        else:
            loss = self.criterion(logits, y.long())
            preds = torch.argmax(logits, dim=1)
            acc = (preds == y).float().mean()
        
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        
        if self.loss_type == 'binary':
            y = y.float().unsqueeze(1)
            loss = self.criterion(logits, y)
            preds = torch.sigmoid(logits) > 0.5
            acc = (preds == y).float().mean()
        else:
            loss = self.criterion(logits, y.long())
            preds = torch.argmax(logits, dim=1)
            acc = (preds == y).float().mean()
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=TRAINING_CONFIG['learning_rate'],
            weight_decay=TRAINING_CONFIG['weight_decay']
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=TRAINING_CONFIG['lr_reduce_factor'],
            patience=TRAINING_CONFIG['patience_lr_reduce'],
            min_lr=TRAINING_CONFIG['min_lr']
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }


class AircraftDataset(Dataset):
    def __init__(self, filenames, df, transform=None):
        self.filenames = filenames
        self.df = df
        self.transform = transform
        
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        filename = self.filenames[idx]
        specs, labels = audio_to_spectrogram(filename, self.df)
        
        if len(specs) == 0:
            # Return dummy data if file processing fails
            dummy_spec = np.zeros(AUDIO_CONFIG['exp_input_shape'])
            return torch.FloatTensor(dummy_spec).unsqueeze(0), torch.tensor(0.0)
        
        # Use first spectrogram segment
        spec = specs[0]
        spec = normalise_array(spec)
        
        # Get aircraft detection label (0 = background, 1 = aircraft)
        label = float(labels['is_aircraft'])
        
        spec_tensor = torch.FloatTensor(spec).unsqueeze(0)  # Add channel dimension
        
        if self.transform:
            spec_tensor = self.transform(spec_tensor)
            
        return spec_tensor, torch.tensor(label)


def get_audio_path_and_labels(df, filename):
    """Get audio file path and all target labels"""
    row = df.loc[df['filename'] == filename]
    if len(row) == 0:
        return None, None
    
    row = row.iloc[0]
    filepath = os.path.join(DATA_CONFIG['audio_dir'], str(row['class']), filename)
    
    labels = {}
    labels['is_aircraft'] = row['class']
    
    if row['class'] == 1:
        labels['engtype'] = row['engtype']
        if row['engnum'] in [1, 2]:
            labels['engnum'] = row['engnum']
        else:
            labels['engnum'] = None
        labels['fueltype'] = row['fueltype']
    else:
        labels['engtype'] = None
        labels['engnum'] = None
        labels['fueltype'] = None
    
    return filepath, labels


def audio_to_spectrogram(filename, df):
    """Convert audio file to mel spectrogram"""
    path, labels = get_audio_path_and_labels(df, filename)
    
    if path is None or labels is None:
        return [], {}
    
    try:
        signal, sr = librosa.load(path)
        
        if sr != AUDIO_CONFIG['sr']:
            raise ValueError('Sample rate mismatch between audio and target')
            
        clip_segments = math.ceil(len(signal) / AUDIO_CONFIG['samples_per_segment'])
        specs = []
        
        for segment in range(clip_segments):
            start = AUDIO_CONFIG['samples_per_segment'] * segment
            end = start + AUDIO_CONFIG['samples_per_segment'] - AUDIO_CONFIG['hop_length']
            
            spec = librosa.feature.melspectrogram(
                y=signal[start:end], 
                sr=sr, 
                n_fft=AUDIO_CONFIG['n_fft'], 
                n_mels=AUDIO_CONFIG['n_mels'], 
                hop_length=AUDIO_CONFIG['hop_length'],
                fmin=AUDIO_CONFIG['fmin'],
                fmax=AUDIO_CONFIG['fmax'],
                window=AUDIO_CONFIG['window']
            )
            
            db_spec = librosa.power_to_db(spec, ref=0.0)
            
            if db_spec.shape[1] == AUDIO_CONFIG['exp_vectors_per_segment']:
                specs.append(db_spec)
            elif db_spec.shape[1] < AUDIO_CONFIG['exp_vectors_per_segment']:
                n_short = AUDIO_CONFIG['exp_vectors_per_segment'] - db_spec.shape[1]
                db_spec = np.pad(db_spec, [(0, 0), (0, n_short)], 'constant')
                specs.append(db_spec)
        
        return specs, labels
    except Exception:
        return [], {}


def normalise_array(array):
    """Normalize array to [0, 1] range"""
    array = np.asarray(array)
    min_val = array.min()
    max_val = array.max()
    if max_val > min_val:
        norm_array = (array - min_val) / (max_val - min_val)
    else:
        norm_array = array
    return norm_array


def create_data_loaders(df):
    """Create train, validation, and test data loaders for aircraft detection"""
    # Split dataset
    train_df = df.loc[(df['fold'] == '1') | (df['fold'] == '2') | (df['fold'] == '3') | (df['fold'] == '4')]
    val_df = df.loc[df['fold'] == '5']
    test_df = df.loc[df['fold'] == 'test']
    
    train_files = train_df['filename'].reset_index(drop=True)
    val_files = val_df['filename'].reset_index(drop=True)
    test_files = test_df['filename'].reset_index(drop=True)
    
    # Data split logging removed for cleaner output
    
    # Create datasets
    train_dataset = AircraftDataset(train_files, df)
    val_dataset = AircraftDataset(val_files, df)
    test_dataset = AircraftDataset(test_files, df)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAINING_CONFIG['batch_size'],
        shuffle=True,
        num_workers=TRAINING_CONFIG['num_workers'],
        pin_memory=TRAINING_CONFIG['pin_memory'],
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=TRAINING_CONFIG['batch_size'],
        shuffle=False,
        num_workers=TRAINING_CONFIG['num_workers'],
        pin_memory=TRAINING_CONFIG['pin_memory']
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=TRAINING_CONFIG['batch_size'],
        shuffle=False,
        num_workers=TRAINING_CONFIG['num_workers'],
        pin_memory=TRAINING_CONFIG['pin_memory']
    )
    
    return train_loader, val_loader, test_loader
