# Aircraft Detection from Audio - Multi-Class Classification

Deep learning system to classify aircraft characteristics from audio recordings using convolutional recurrent neural networks (CRNNs)

## Project Overview


### Classification
- **Aircraft Detection** (Binary: aircraft vs no aircraft)
- **Engine Type** (4 classes: Turbofan, Turboprop, Piston, Turboshaft)
- **Engine Number** (3 classes: 1, 2, 4 engines)
- **Fuel Type** (2 classes: Kerosene, Gasoline)

### Two Training Approaches
1. **Single Multi-Output Model** (`train_single_model.py`) - One model predicts all 4 tasks
2. **Multiple Separate Models** (`train_several_model.py`) - Four individual models for each task

## Dataset Analysis

![Dataset Analysis](dataset_analysis.png)

### Dataset Structure
- **Total Recordings**: 1,895 audio files
- **Aircraft Recordings**: 625 (class=1)
- **No Aircraft Recordings**: 1,270 (class=0)
- **Audio Format**: WAV files
- **Duration**: Variable (typically 10-60 seconds)

### Target Classification Variables

#### 1. Engine Type (engtype) - 4 Classes
| Class | Count | Percentage | Balance Ratio |
|-------|-------|------------|---------------|
| Turbofan | 470 | 75.2% | 117.5:1 |
| Turboprop | 114 | 18.2% | 29.3:1 |
| Piston | 37 | 5.9% | 9.3:1 |
| Turboshaft | 4 | 0.6% | 1:1 |

#### 2. Engine Number (engnum) - 3 Classes
| Class | Count | Percentage | Balance Ratio |
|-------|-------|------------|---------------|
| 2 engines | 583 | 93.3% | 194.3:1 |
| 1 engine | 39 | 6.2% | 15.0:1 |
| 4 engines | 3 | 0.5% | 1:1 |

#### 3. Fuel Type (fueltype) - 2 Classes
| Class | Count | Percentage | Balance Ratio |
|-------|-------|------------|---------------|
| Kerosene | 588 | 94.1% | 15.9:1 |
| Gasoline | 37 | 5.9% | 1:1 |

### Data Quality Assessment
- **Missing Data**: All target variables have complete data for aircraft recordings (class=1)
- **Data Integrity**: No missing values in aircraft recordings for classification targets
- **Class Balance**: Highly imbalanced dataset requiring special handling strategies

## Model Architecture Comparison

### Approach Comparison

| Feature | Single Multi-Output Model | Multiple Separate Models |
|---------|---------------------------|-------------------------|
| **Script** | `train_single_model.py` | `train_several_model.py` |
| **Models Count** | 1 model with 4 outputs | 4 separate models |
| **Parameters** | ~24M total | ~96M total (24M × 4) |
| **Training Time** | 1× training session | 4× training sessions |
| **Inference Speed** | 1× prediction call | 4× prediction calls |
| **Memory Usage** | Lower | Higher |
| **Feature Sharing** | Shared CNN backbone | Independent CNNs |
| **Knowledge Transfer** | Cross-task learning | No task interaction |
| **Model Files** | 1 file (.h5) | 4 files (.h5 each) |
| **Best For** | Related tasks, limited data | Independent tasks, abundant data |

### Single Multi-Output Model Architecture

```
Input: Audio Spectrogram (128, 107, 1)
    ↓
┌─────────────────────────────────────┐
│        SHARED CNN BACKBONE          │
│  Conv2D(32) → BatchNorm → MaxPool   │
│  Conv2D(64) → BatchNorm → MaxPool   │
│  Conv2D(128) → BatchNorm → MaxPool  │
│  Flatten → Dense(256) → Dropout     │
└─────────────────────────────────────┘
    ↓
┌─────────────┬─────────────┬─────────────┬─────────────┐
│ is_aircraft │   engtype   │   engnum    │  fueltype   │
│ Dense(1)    │ Dense(4)    │ Dense(3)    │ Dense(2)    │
│ sigmoid     │ softmax     │ softmax     │ softmax     │
└─────────────┴─────────────┴─────────────┴─────────────┘
     Binary        4 classes    3 classes    2 classes
```

### Multiple Separate Models Architecture

```
Model 1: Aircraft Detection
Input: Spectrogram (128, 107, 1) → CNN → Dense → sigmoid (1 output)

Model 2: Engine Type
Input: Spectrogram (128, 107, 1) → CNN → Dense → softmax (4 outputs)

Model 3: Engine Number  
Input: Spectrogram (128, 107, 1) → CNN → Dense → softmax (3 outputs)

Model 4: Fuel Type
Input: Spectrogram (128, 107, 1) → CNN → Dense → softmax (2 outputs)
```

## Technical Implementation

### Audio Processing Pipeline
1. **Audio Loading**: Using librosa for audio file loading and resampling
2. **Segmentation**: Extract fixed-duration segments from variable-length recordings
3. **Feature Extraction**: Convert audio to mel-spectrograms using librosa
4. **Normalization**: Apply power-to-decibel conversion for better model training

### Model Architecture
- **Base Model**: Convolutional Neural Network
- **Input**: Mel-spectrogram images (time-frequency representations)
- **Layers**: Conv2D, MaxPooling2D, Dropout, BatchNormalization
- **Output**: Multi-class classification with softmax activation

### Training Strategy
- **Data Split**: Stratified train/validation/test split to maintain class proportions
- **Class Weights**: Applied to handle imbalanced class distributions
- **Masked Loss**: For multi-output model, ignore non-aircraft samples in multi-class tasks
- **Regularization**: Dropout and batch normalization to prevent overfitting

## Project Structure

```
drone-detection/
├── dataset/
│   ├── audio/                 # Audio files directory
│   ├── env_audio/            # Environment audio files
│   ├── sample_meta.csv       # Metadata with aircraft information
│   ├── environment_mappings_raw.csv
│   └── environment_class_mappings.csv
├── train_single_model.py      # Single multi-output model training
├── train_several_model.py     # Multiple separate models training
├── analyze_dataset.py         # Dataset analysis script
├── get_data.py               # Data download script
├── model_comparison.md        # Detailed architecture comparison
├── README.md                 # This file
├── requirements.txt          # Python dependencies
└── venv/                    # Virtual environment
```

