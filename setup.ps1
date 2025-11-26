# PowerShell setup script for LockerLink AI microservice

Write-Host "Setting up LockerLink AI microservice..." -ForegroundColor Green

# Check if SAM3 directory exists
if (-not (Test-Path "sam3")) {
    Write-Host "Cloning SAM3 repository..." -ForegroundColor Yellow
    git clone https://github.com/facebookresearch/sam3.git sam3
} else {
    Write-Host "SAM3 directory already exists, skipping clone..." -ForegroundColor Green
}

# Check Python version
$pythonVersion = python --version 2>&1 | ForEach-Object { $_.ToString().Split(' ')[1] }
$requiredVersion = "3.12"

Write-Host "Python version: $pythonVersion" -ForegroundColor Cyan

# Install PyTorch with CUDA support (if CUDA is available)
try {
    $nvcc = Get-Command nvcc -ErrorAction Stop
    Write-Host "CUDA detected, installing PyTorch with CUDA 12.6 support..." -ForegroundColor Yellow
    pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
} catch {
    Write-Host "CUDA not detected, installing CPU-only PyTorch..." -ForegroundColor Yellow
    Write-Host "Warning: Performance will be limited. For production, use CUDA 12.6+" -ForegroundColor Red
    pip install torch==2.7.0 torchvision torchaudio
}

# Install other dependencies
Write-Host "Installing dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt

# Install SAM3
Write-Host "Installing SAM3..." -ForegroundColor Yellow
pip install -e ./sam3

# Check for checkpoints
$checkpointFiles = Get-ChildItem -Path "models/sam3_weights" -Filter "*.pth" -ErrorAction SilentlyContinue
if ($null -eq $checkpointFiles -or $checkpointFiles.Count -eq 0) {
    Write-Host ""
    Write-Host "⚠️  WARNING: SAM3 checkpoints not found in models/sam3_weights/" -ForegroundColor Red
    Write-Host "Please:" -ForegroundColor Yellow
    Write-Host "1. Request access at https://huggingface.co/facebook/sam3" -ForegroundColor Yellow
    Write-Host "2. Download the checkpoint files" -ForegroundColor Yellow
    Write-Host "3. Place them in models/sam3_weights/" -ForegroundColor Yellow
} else {
    Write-Host "✓ SAM3 checkpoints found" -ForegroundColor Green
}

Write-Host ""
Write-Host "Setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "To run the service:" -ForegroundColor Cyan
Write-Host "  uvicorn app.main:app --host 0.0.0.0 --port 8000" -ForegroundColor White
Write-Host ""
Write-Host "To test:" -ForegroundColor Cyan
Write-Host "  curl http://localhost:8000/test" -ForegroundColor White

