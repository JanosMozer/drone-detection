# Aircraft Detection from Audio - Multi-Class Classification

Deep learning system to classify aircraft characteristics from audio recordings using specialized convolutional neural networks (CNNs)

## Project Overview

This project implements aircraft classification from audio recordings using **four separate specialized models**, each trained for a specific classification task with GPU acceleration support.

### Classification Tasks
- **Aircraft Detection** (Binary: aircraft vs no aircraft)
- **Engine Type** (4 classes: Turbofan, Turboprop, Piston, Turboshaft)
- **Engine Number** (Binary: 1 or 2 engines, excluding 4-engine aircraft)
- **Fuel Type** (Binary: Kerosene or Gasoline) 

### Training Approach
**Multiple Separate Models** - Four independent CNN models, each specialized for one classification task, trained in sequence: binary models first, then multi-class models. Available in both CPU (`train_model.py`) and GPU-optimized (`train_models_CUDA.py`) versions.

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

#### 2. Engine Number (engnum) - Binary Classification
| Class | Count | Percentage | Balance Ratio |
|-------|-------|------------|---------------|
| 2 engines | 583 | 93.7% | 14.9:1 |
| 1 engine | 39 | 6.3% | 1:1 |
| 4 engines | 3 | **Excluded** | - |

*Note: 4-engine aircraft are excluded from this classification task due to extremely low sample count.*

#### 3. Fuel Type (fueltype) - 2 Classes
| Class | Count | Percentage | Balance Ratio |
|-------|-------|------------|---------------|
| Kerosene | 588 | 94.1% | 15.9:1 |
| Gasoline | 37 | 5.9% | 1:1 |

### Data Quality Assessment
- **Missing Data**: All target variables have complete data for aircraft recordings (class=1)
- **Data Integrity**: No missing values in aircraft recordings for classification targets
- **Class Balance**: Highly imbalanced dataset requiring special handling strategies

## Model Architecture

This project uses **four separate specialized CNN models**, each optimized for a specific classification task. This approach allows each model to focus on task-specific features without interference from other classification objectives.

### Architecture Overview

Each model follows the same CNN architecture but is trained independently:

```
Input: Audio Spectrogram (128, 107, 1)
    ↓
┌─────────────────────────────────────┐
│           CNN BACKBONE              │
│  Conv2D(32) → BatchNorm → MaxPool   │
│  Conv2D(64) → BatchNorm → MaxPool   │
│  Conv2D(128) → BatchNorm → MaxPool  │
│  Flatten → Dense(256) → Dropout     │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│         TASK-SPECIFIC OUTPUT        │
│    Dense(n) → Activation            │
└─────────────────────────────────────┘
```

### Individual Model Specifications

| Model | Task | Output Classes | Activation | Data Filter |
|-------|------|----------------|------------|-------------|
| **Model 1** | Aircraft Detection | 1 (Binary) | Sigmoid | All samples |
| **Model 2** | Engine Number | 1 (Binary) | Sigmoid | 1-2 engine aircraft only |
| **Model 3** | Fuel Type | 1 (Binary) | Sigmoid | Aircraft samples only |
| **Model 4** | Engine Type | 4 (Multi-class) | Softmax | Aircraft samples only |

### Training Sequence

Models are trained in a specific order to optimize training efficiency:

1. **Binary Models First** (faster convergence):
   - Aircraft Detection (all data)
   - Engine Number (filtered data)
   - Fuel Type (aircraft only)

2. **Multi-class Model Last**:
   - Engine Type (aircraft only, most complex)

### GPU Optimization Features

The `train_models_CUDA.py` version includes:
- **Mixed Precision Training** (FP16/FP32) for 2× speed improvement
- **Optimized Batch Sizes** (64 vs 32 for CPU)
- **tf.data Pipeline** with prefetching for efficient GPU utilization
- **Memory Growth** to prevent OOM errors
- **Enhanced Architecture** with additional layers for better GPU utilization

## Results and Outputs

After training, all results are automatically saved in the `results/` directory:

### **Training Outputs**
- **Model Files**: Trained model weights and label encoders
- **Training History**: Complete training metrics for all epochs
- **Processed Data**: Preprocessed spectrograms and encoded labels
- **Visualizations**: Training charts showing loss and accuracy curves
- **Performance Summary**: JSON file with final test accuracies and model stats

### **Training Charts**
The `training_charts.png` file contains 8 subplots showing:
- **Loss curves** for each task (training vs validation)
- **Accuracy curves** for each task (training vs validation)
- **Tasks**: Aircraft Detection, Engine Type, Engine Number, Fuel Type

### **Data Persistence**
All preprocessed data is saved for:
- **Reproducibility**: Exact same training conditions
- **Analysis**: Post-training performance investigation
- **Transfer Learning**: Reuse processed data for new models
- **Debugging**: Investigate model behavior on specific samples

## Technical Implementation

### Audio Processing Pipeline
1. **Audio Loading**: Using librosa for audio file loading and resampling
2. **Segmentation**: Extract fixed-duration segments from variable-length recordings
3. **Feature Extraction**: Convert audio to mel-spectrograms using librosa
4. **Normalization**: Apply power-to-decibel conversion for better model training

### Model Architecture
- **Base Models**: Four separate Convolutional Neural Networks
- **Input**: Mel-spectrogram images (128×107×1 time-frequency representations)
- **Layers**: Conv2D, MaxPooling2D, Dropout, BatchNormalization, Dense
- **Outputs**: Task-specific (Binary: sigmoid, Multi-class: softmax)
- **Specialization**: Each model optimized for one specific classification task

### Training Strategy
- **Data Split**: Stratified train/validation/test split to maintain class proportions
- **Class Weights**: Applied to handle imbalanced class distributions  
- **Masked Loss Functions**: Custom loss functions ignore invalid samples (e.g., non-aircraft for engine classification)
- **Sequential Training**: Binary models first (faster), then multi-class models
- **Task-Specific Filtering**: Each model trained only on relevant data samples
- **Regularization**: Dropout and batch normalization to prevent overfitting
- **GPU Acceleration**: Mixed precision training and optimized data pipelines for faster training

## Project Structure

```
drone-detection/
├── dataset/
│   ├── audio/                 # Audio files directory
│   ├── env_audio/            # Environment audio files
│   ├── sample_meta.csv       # Metadata with aircraft information
│   ├── environment_mappings_raw.csv
│   └── environment_class_mappings.csv
├── train_model.py             # CPU-optimized separate models training
├── train_models_CUDA.py       # GPU-optimized separate models training
├── evaluation_utils.py        # Modular evaluation and results functions
├── analyze_dataset.py         # Dataset analysis script
├── get_data.py               # Data download script
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── venv/                    # Virtual environment
└── results/                  # Training results and outputs
    ├── is_aircraft_model.h5           # Aircraft detection model
    ├── engtype_model.h5               # Engine type classification model
    ├── engnum_model.h5                # Engine number classification model
    ├── fueltype_model.h5              # Fuel type classification model
    ├── label_encoders.pkl             # Label encoders for all tasks
    ├── training_histories.pkl         # Training metrics for all models
    ├── training_charts_separate_models.png  # Training visualizations
    ├── performance_results_separate_models.json  # Performance summary
    ├── X_train.npy                   # Processed training data
    ├── X_val.npy                     # Processed validation data
    ├── X_test.npy                    # Processed test data
    ├── y_train.pkl                   # Training labels
    ├── y_val.pkl                     # Validation labels
    └── y_test.pkl                    # Test labels
```

