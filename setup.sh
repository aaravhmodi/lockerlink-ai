#!/bin/bash
# Setup script for LockerLink AI microservice

set -e

echo "Setting up LockerLink AI microservice..."

# Check if SAM3 directory exists
if [ ! -d "sam3" ]; then
    echo "Cloning SAM3 repository..."
    git clone https://github.com/facebookresearch/sam3.git sam3
else
    echo "SAM3 directory already exists, skipping clone..."
fi

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.12"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "Warning: Python 3.12+ is required. Found: $python_version"
    echo "Please upgrade Python or use a virtual environment with Python 3.12+"
fi

# Install PyTorch with CUDA support (if CUDA is available)
if command -v nvcc &> /dev/null; then
    echo "CUDA detected, installing PyTorch with CUDA 12.6 support..."
    pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
else
    echo "CUDA not detected, installing CPU-only PyTorch..."
    echo "Warning: Performance will be limited. For production, use CUDA 12.6+"
    pip install torch==2.7.0 torchvision torchaudio
fi

# Install other dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Install SAM3
echo "Installing SAM3..."
pip install -e ./sam3

# Check for checkpoints
if [ ! -d "models/sam3_weights" ] || [ -z "$(ls -A models/sam3_weights/*.pth models/sam3_weights/*.pt 2>/dev/null)" ]; then
    echo ""
    echo "⚠️  WARNING: SAM3 checkpoints not found in models/sam3_weights/"
    echo "Please:"
    echo "1. Request access at https://huggingface.co/facebook/sam3"
    echo "2. Download the checkpoint files"
    echo "3. Place them in models/sam3_weights/"
else
    echo "✓ SAM3 checkpoints found"
fi

echo ""
echo "Setup complete!"
echo ""
echo "To run the service:"
echo "  uvicorn app.main:app --host 0.0.0.0 --port 8000"
echo ""
echo "To test:"
echo "  curl http://localhost:8000/test"

