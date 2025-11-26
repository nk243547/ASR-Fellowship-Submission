#!/usr/bin/env bash

set -e

echo "=== ASR Fellowship Adapter Pipeline (CPU) ==="

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 1. Create virtual environment only if missing
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "Virtual environment created"
    
    # Activate venv
    source venv/bin/activate
    
    # Upgrade pip
    echo "Upgrading pip..."
    pip install --upgrade pip
    
    # Install dependencies (only on first run)
    echo "Installing dependencies (first time setup)..."
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
    pip install transformers datasets jiwer soundfile librosa
    echo "All packages installed successfully!"
else
    echo "Virtual environment already exists, skipping dependency installation"
    # Activate existing venv
    source venv/bin/activate
fi

# Add current directory to Python path for the module imports
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

# 2. Download dataset only if not exists
if [ ! -d "data/raw" ] || [ -z "$(ls -A data/raw 2>/dev/null)" ]; then
    echo "Downloading dataset..."
    python src/data_prep.py
    echo "Dataset downloaded"
else
    echo "Dataset already exists, skipping download"
fi

# 3. Train adapters only if no checkpoints exist
if [ ! -d "checkpoints/adapter" ] || [ -z "$(ls -A checkpoints/adapter/*.pt 2>/dev/null)" ]; then
    echo "Starting training..."
    python src/train_adapter.py
    echo "Training completed"
else
    echo "Checkpoints already exist, skipping training"
    echo "Use './run.sh --retrain' to force retraining"
fi

# 4. Always run evaluation (it's fast)
echo "Running evaluation..."
python src/eval.py

echo "=== Pipeline complete ==="
echo "Virtual environment is still active. To deactivate, run: deactivate"
echo "To resume later, just run: ./run.sh"