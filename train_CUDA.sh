#!/bin/bash

# This isolates TensorFlow's bundled CUDA from system CUDA 13.0

echo "Setting up clean CUDA environment for TensorFlow..."
export LD_LIBRARY_PATH_BACKUP="$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH=$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -v cuda | tr '\n' ':' | sed 's/:$//')

unset CUDA_HOME
unset CUDA_ROOT
unset CUDA_PATH

export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_GPU_ALLOCATOR=cuda_malloc_async

echo "Environment cleaned. Starting training..."
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

source /mnt/HDD/drone-detection/venv/bin/activate
python /mnt/HDD/drone-detection/train_models_CUDA.py

export LD_LIBRARY_PATH="$LD_LIBRARY_PATH_BACKUP"
echo "Training completed. Environment restored."
