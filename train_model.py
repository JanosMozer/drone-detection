# import required libraries
import os
import math
import random
import pandas as pd
import numpy as np
import IPython.display as ipd
import librosa
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Dropout, BatchNormalization, SpatialDropout2D
from tensorflow.keras import regularizers
from sklearn.metrics import average_precision_score, PrecisionRecallDisplay
from sklearn.utils import class_weight
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from evaluation_utils import save_separate_models_results, evaluate_separate_models

tf.get_logger().setLevel('ERROR')

# Load the dataset
df = pd.read_csv('dataset/sample_meta.csv')
print(f"Dataset loaded: {len(df)} samples")

# Define audio directory
AUDIO_DIR = 'dataset/audio/audio'

# Multi-class classification setup - we'll train 4 models
TARGET_COLUMNS = {
    'is_aircraft': 'class',  # Binary: aircraft vs no aircraft
    'engtype': 'engtype',    # 4 classes: Turbofan, Turboprop, Piston, Turboshaft
    'engnum': 'engnum',      # Binary: 1 or 2 engines (excluding 4-engine aircraft)
    'fueltype': 'fueltype'   # Binary: Kerosene or Gasoline
}

# Multiple separate models setup - train 4 independent models
print("Training 4 separate models, one for each task")

def get_audio_path_and_labels(df, filename):
    """Get audio file path and all target labels for multi-output classification"""
    row = df.loc[df['filename'] == filename]
    if len(row) == 0:
        return None, None
    
    row = row.iloc[0]
    
    # Get file path based on aircraft class
    filepath = os.path.join(AUDIO_DIR, str(row['class']), filename)
    
    # Get all target labels
    labels = {}
    labels['is_aircraft'] = row['class']
    
    # For aircraft recordings, get additional labels
    if row['class'] == 1:
        labels['engtype'] = row['engtype']
        # For engnum, only include 1 or 2 engines (exclude 4-engine aircraft)
        if row['engnum'] in [1, 2]:
            labels['engnum'] = row['engnum']
        else:
            labels['engnum'] = None  # Exclude 4-engine aircraft
        labels['fueltype'] = row['fueltype']
    else:
        # For non-aircraft, set other labels to None
        labels['engtype'] = None
        labels['engnum'] = None
        labels['fueltype'] = None
    
    return filepath, labels

# function to load a file to play and show it's waveform
def load_show_audio(filename):
    path, labels = get_audio_path_and_labels(df, filename)
    if path is None:
        return None
    signal, sr = librosa.load(path)
    plt.figure(figsize=(6, 3))
    librosa.display.waveshow(y=signal, sr=sr)
    plt.show()
    return ipd.Audio(path)

# set some constants for feature extraction, training and inference
SR = 22050 # sample rate of the audio files
DURATION = 5 # length of a segment in seconds
SAMPLES_PER_SEGMENT = SR*DURATION # the number of samples per segment we expect
N_FFT = 2048 # approx frequency resolution of 21.5 Hz
HOP_LENGTH = 1024 
EXP_VECTORS_PER_SEGMENT = math.floor(SAMPLES_PER_SEGMENT/HOP_LENGTH)
N_MELS = 128 # the number of frequency bins for spectrogram
EXP_INPUT_SHAPE = (N_MELS, EXP_VECTORS_PER_SEGMENT) # the expected shape of the spectrogram
print('Expected spectrogram shape:', EXP_INPUT_SHAPE)

# function to load a file and chop it into spectrograms equal to the segment length
def audio_to_spectrogram(filename):
    path, labels = get_audio_path_and_labels(df, filename)
    
    # Skip if this is not a valid file
    if path is None or labels is None:
        return [], {}
    
    signal, sr = librosa.load(path)

    
    if sr != SR:
        raise ValueError('Sample rate mismatch between audio and target')
        
    clip_segments = math.ceil(len(signal) / SAMPLES_PER_SEGMENT)
    
    # empty list to hold the spectrograms for this clip
    specs = []
    
    for segment in range(clip_segments):
        
        start = SAMPLES_PER_SEGMENT * segment
        end = start + SAMPLES_PER_SEGMENT - HOP_LENGTH
        
        spec = librosa.feature.melspectrogram(y=signal[start:end], 
                                              sr=sr, n_fft=N_FFT, 
                                              n_mels=N_MELS, 
                                              hop_length=HOP_LENGTH,
                                              window='hann')
        
        db_spec = librosa.power_to_db(spec, ref=0.0)
        
        if db_spec.shape[1] == EXP_VECTORS_PER_SEGMENT:
            specs.append(db_spec)
        
        # if the clip is shorter than the segment, add zero padding to the right
        elif db_spec.shape[1] < EXP_VECTORS_PER_SEGMENT:
            n_short = EXP_VECTORS_PER_SEGMENT - db_spec.shape[1]
            db_spec = np.pad(db_spec, [(0, 0), (0, n_short)], 'constant')
            specs.append(db_spec)
        
    return specs, labels

    # function to apply min-max scaling to squeeze spectrogram values between 0 and 1
def normalise_array(array):
    array = np.asarray(array)
    min_val = array.min()
    max_val = array.max()
    
    norm_array = (array - min_val) / (max_val - min_val)
    
    return norm_array

    # wrapper function to take a list of files and extract their features 
# -> array of features (X) and dictionary of corresponding labels (y_dict)
def preprocess(file_list):
    data = {
        'feature': [],
        'is_aircraft': [],
        'engtype': [],
        'engnum': [],
        'fueltype': []
    }
    
    for file in file_list:
        specs, labels = audio_to_spectrogram(filename=file)
        
        # Skip files that couldn't be processed
        if len(specs) == 0:
            continue
        
        for spec in specs:
            norm_spec = normalise_array(spec)
            data['feature'].append(norm_spec)
            data['is_aircraft'].append(labels['is_aircraft'])
            data['engtype'].append(labels['engtype'])
            data['engnum'].append(labels['engnum'])
            data['fueltype'].append(labels['fueltype'])
    
    X = np.asarray(data['feature'])
    y_dict = {
        'is_aircraft': np.asarray(data['is_aircraft']),
        'engtype': np.asarray(data['engtype']),
        'engnum': np.asarray(data['engnum']),
        'fueltype': np.asarray(data['fueltype'])
    }
    
    return X, y_dict

# Prepare label encoders for all tasks
label_encoders = {}
class_weights = {}

# is_aircraft: binary, no encoding needed
class_weights['is_aircraft'] = {0: 1.0, 1: 2.03}  # Balanced weights

# engtype: 4 classes
aircraft_df = df[df['class'] == 1]
le_engtype = LabelEncoder()
le_engtype.fit(aircraft_df['engtype'])
label_encoders['engtype'] = le_engtype
class_weights['engtype'] = {0: 4.22, 1: 0.33, 2: 1.37, 3: 39.06}  # Pre-computed

# engnum: binary (1 or 2 engines, excluding 4-engine aircraft)
engnum_filtered = aircraft_df[aircraft_df['engnum'].isin([1, 2])]
le_engnum = LabelEncoder()
le_engnum.fit(engnum_filtered['engnum'])
label_encoders['engnum'] = le_engnum
class_weights['engnum'] = {0: 5.34, 1: 0.36}  # Only for 1 and 2 engines

# fueltype: binary (Kerosene or Gasoline)
le_fueltype = LabelEncoder()
le_fueltype.fit(aircraft_df['fueltype'])
label_encoders['fueltype'] = le_fueltype
class_weights['fueltype'] = {0: 8.45, 1: 0.53}  # Binary classification

# Split dataset - use all recordings
train_df = df.loc[(df['fold'] == '1') | (df['fold'] == '2') | (df['fold'] == '3') | (df['fold'] == '4')]
val_df = df.loc[df['fold'] == '5']
test_df = df.loc[df['fold'] == 'test']

train = train_df['filename'].reset_index(drop=True)
val = val_df['filename'].reset_index(drop=True)
test = test_df['filename'].reset_index(drop=True)

print(f'TRAIN: {len(train)} files, VAL: {len(val)} files, TEST: {len(test)} files')

# Preprocess the datasets
print("Preprocessing...")
X_train, y_train_dict = preprocess(train)
X_val, y_val_dict = preprocess(val)
X_test, y_test_dict = preprocess(test)

# Encode labels for multi-class tasks
def encode_labels(y_dict, encoders):
    encoded = {}
    
    # is_aircraft: no encoding needed
    encoded['is_aircraft'] = y_dict['is_aircraft']
    
    # For other tasks, only encode aircraft recordings
    aircraft_mask = y_dict['is_aircraft'] == 1
    
    for task in ['engtype', 'engnum', 'fueltype']:
        # Filter out None values for engnum (4-engine aircraft)
        if task == 'engnum':
            valid_mask = (y_dict[task] != None) & aircraft_mask
            task_labels = y_dict[task][valid_mask]
        else:
            task_labels = y_dict[task][aircraft_mask]
            valid_mask = aircraft_mask
        
        if len(task_labels) > 0:
            encoded_labels = encoders[task].transform(task_labels)
            
            # Create full array with -1 for non-aircraft or invalid samples
            full_encoded = np.full(len(y_dict['is_aircraft']), -1, dtype=int)
            full_encoded[valid_mask] = encoded_labels
            encoded[task] = full_encoded
        else:
            # If no valid labels, create array of all -1s
            encoded[task] = np.full(len(y_dict['is_aircraft']), -1, dtype=int)
    
    return encoded

y_train = encode_labels(y_train_dict, label_encoders)
y_val = encode_labels(y_val_dict, label_encoders)
y_test = encode_labels(y_test_dict, label_encoders)

# Add channel dimension for CNN
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], X_val.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

print(f'X_train: {X_train.shape}')

# Build individual CNN model for single task
def build_single_task_model(input_shape, task_name, num_classes, activation='sigmoid'):
    """Build CNN model for a single classification task"""
    from tensorflow.keras.layers import Input
    from tensorflow.keras.models import Model
    
    inputs = Input(shape=input_shape)
    
    x = Conv2D(32, (3, 3), activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)
    
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)
    
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)
    
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    # Output layer for the specific task
    output = Dense(num_classes, activation=activation, name=task_name)(x)
    
    model = Model(inputs=inputs, outputs=output)
    
    return model

# Custom loss function to ignore non-aircraft samples for multi-class tasks
def masked_sparse_categorical_crossentropy(y_true, y_pred):
    """Ignore samples with label -1"""
    mask = tf.cast(tf.not_equal(y_true, -1), tf.float32)
    y_true_masked = tf.maximum(y_true, 0)  # Replace -1 with 0 for calculation
    loss = tf.keras.losses.sparse_categorical_crossentropy(y_true_masked, y_pred)
    return tf.reduce_sum(loss * mask) / tf.maximum(tf.reduce_sum(mask), 1.0)

# Custom loss function for binary classification with masked labels
def masked_binary_crossentropy(y_true, y_pred):
    """Ignore samples with label -1 for binary classification"""
    mask = tf.cast(tf.not_equal(y_true, -1), tf.float32)
    y_true_masked = tf.maximum(y_true, 0)  # Replace -1 with 0 for calculation
    y_true_masked = tf.cast(y_true_masked, tf.float32)
    loss = tf.keras.losses.binary_crossentropy(y_true_masked, y_pred)
    return tf.reduce_sum(loss * mask) / tf.maximum(tf.reduce_sum(mask), 1.0)

# Train 4 separate models - binary models first, then engtype
input_shape = X_train.shape[1:]
models = {}
histories = {}

# Define model configurations - binary models first, engtype last
model_configs = [
    # Binary models first
    {
        'name': 'is_aircraft',
        'num_classes': 1,
        'activation': 'sigmoid',
        'loss': 'binary_crossentropy',
        'data_filter': None  # Use all data
    },
    {
        'name': 'engnum',
        'num_classes': 1,
        'activation': 'sigmoid',
        'loss': masked_binary_crossentropy,
        'data_filter': 'engnum_valid'  # Only 1 or 2 engine aircraft
    },
    {
        'name': 'fueltype',
        'num_classes': 1,
        'activation': 'sigmoid',
        'loss': masked_binary_crossentropy,
        'data_filter': 'aircraft_only'  # Only aircraft data
    },
    # Multi-class model last
    {
        'name': 'engtype',
        'num_classes': 4,
        'activation': 'softmax',
        'loss': masked_sparse_categorical_crossentropy,
        'data_filter': 'aircraft_only'  # Only aircraft data
    }
]

# Train each model separately
for config in model_configs:
    task_name = config['name']
    print(f"\n{'='*50}")
    print(f"Training {task_name.upper()} model")
    print(f"{'='*50}")
    
    # Build model for this task
    model = build_single_task_model(
        input_shape, 
        task_name, 
        config['num_classes'], 
        config['activation']
    )
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss=config['loss'],
        metrics=['accuracy']
    )
    
    print(f"\n{task_name.upper()} Model Architecture:")
    model.summary()
    
    # Prepare data for this specific task
    if config['data_filter'] == 'engnum_valid':
        # For engnum, exclude samples with -1 (4-engine aircraft)
        valid_mask_train = y_train[task_name] != -1
        valid_mask_val = y_val[task_name] != -1
        
        X_task_train = X_train[valid_mask_train]
        y_task_train = y_train[task_name][valid_mask_train]
        X_task_val = X_val[valid_mask_val] 
        y_task_val = y_val[task_name][valid_mask_val]
    elif config['data_filter'] == 'aircraft_only':
        # For engtype and fueltype, use aircraft samples
        valid_mask_train = y_train['is_aircraft'] == 1
        valid_mask_val = y_val['is_aircraft'] == 1
        
        X_task_train = X_train[valid_mask_train]
        y_task_train = y_train[task_name][valid_mask_train]
        X_task_val = X_val[valid_mask_val] 
        y_task_val = y_val[task_name][valid_mask_val]
    else:
        # Use all data (for is_aircraft)
        X_task_train = X_train
        y_task_train = y_train[task_name]
        X_task_val = X_val
        y_task_val = y_val[task_name]
    
    print(f"\nTraining data shape: {X_task_train.shape}")
    print(f"Training labels shape: {y_task_train.shape}")
    
    # Train the model
    print(f"\nTraining {task_name} model...")
    history = model.fit(
        X_task_train,
        y_task_train,
        validation_data=(X_task_val, y_task_val),
        epochs=30,
        batch_size=32,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True, verbose=1),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=4, verbose=1)
        ],
        verbose=1
    )
    
    # Store model and history
    models[task_name] = model
    histories[task_name] = history
    
    print(f"\n{task_name.upper()} model training completed!")

print(f"\n{'='*60}")
print("All 4 models trained successfully!")
print(f"{'='*60}")

# Evaluate all models and save results
performance_results, all_predictions = evaluate_separate_models(models, X_test, y_test)

# Save all results using evaluation utilities
save_separate_models_results(models, histories, label_encoders, X_train, X_val, X_test, 
                            y_train, y_val, y_test, performance_results)

print("\nTraining completed! 4 separate models trained for each task.")