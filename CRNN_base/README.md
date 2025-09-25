# CRNN Baseline for Aircraft Detection

A PyTorch Lightning implementation of a Convolutional Recurrent Neural Network (CRNN) with attention mechanisms for binary aircraft detection from audio recordings.

## Architecture

### Model Components
- **CNN Frontend**: 4 convolutional blocks (64→128→256→512 channels) with BatchNorm, ReLU, MaxPool, and Dropout
- **CBAM Attention**: Convolutional Block Attention Module after conv blocks 3 and 4
- **Temporal Module**: 2-layer bidirectional GRU (256 hidden units)
- **Attention Pooling**: Multi-head attention (8 heads) over temporal dimension
- **Classification Head**: Dense layers with dropout for binary classification

### Input Specifications
- Mel-spectrogram: 96 mel bins, log amplitude
- Frequency range: 20Hz - 10kHz
- Hop length: 512 samples
- Duration: 5 seconds per segment
- Standardization: Per-recording normalization
- Input shape: (1, 96, 215)


## Task

Binary classification for aircraft detection:
- **Class 0**: Background/Non-aircraft (1270 samples)
- **Class 1**: Aircraft (625 samples)
- **Class Imbalance**: Handled with weighted loss (weight ratio 2.03:1)

## Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Set up Weights & Biases (optional)
export WANDB_API_KEY=your_api_key_here

# Run training
python CRNN_main.py
```

## Configuration

All parameters are centralized in `CRNN_config.py`:
- Audio preprocessing parameters
- Model architecture settings
- Training hyperparameters
- Wandb configuration

## Training Details

- **Optimizer**: AdamW with learning rate 1e-4, weight decay 1e-5
- **Scheduler**: ReduceLROnPlateau (factor=0.5, patience=8)
- **Batch Size**: 64
- **Early Stopping**: 15 epochs patience
- **Loss Function**: Binary Cross Entropy with class weighting
- **Logging**: Weights & Biases integration
- **Data Split**: Fold-based (folds 1-4: train, fold 5: val, test: test)

## Output

Results are saved to timestamped directories containing:
- Model checkpoints (best_aircraft_detection_model.ckpt)
- Wandb logs and metrics
- Training progress tracking
