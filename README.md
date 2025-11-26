# LockerLink AI - Volleyball Video Analysis Microservice

A Python microservice using FastAPI, PyTorch, and SAM3 (Segment Anything Model 3) for analyzing volleyball highlight videos. The service segments players and balls to provide metrics like vertical jump, ball touch, approach speed, kill accuracy, and more.

## Requirements

- Python 3.12+
- PyTorch 2.7.0+ with CUDA 12.6+ support
- CUDA-capable GPU (recommended for performance)
- SAM3 model weights from HuggingFace

## Project Structure

```
lockerlink-ai/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI entry point with /test endpoint
│   ├── sam3_inference.py    # SAM3 model loading and inference
│   └── video_utils.py       # Video download and frame extraction
├── models/
│   └── sam3_weights/        # SAM3 checkpoints (manually downloaded)
├── sam3/                    # SAM3 repository (cloned from GitHub)
├── requirements.txt
├── Dockerfile
└── README.md
```

## Setup Instructions

**We use HuggingFace Transformers to load SAM3** - no need to clone the SAM3 repo!

### 1. Authenticate with HuggingFace

Since you have access to SAM3, authenticate to enable auto-download:

```bash
pip install huggingface_hub
huggingface-cli login
```

Enter your HuggingFace access token (get it from https://huggingface.co/settings/tokens)

**Note**: The model will automatically download from HuggingFace on first use. No manual download needed!

#### Option A: Standard Installation (with CUDA)

```bash
# Install PyTorch with CUDA 12.6 support
pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Install all dependencies (includes transformers for SAM3)
pip install -r requirements.txt
```

#### Option B: CPU-only Installation (not recommended for production)

```bash
pip install torch==2.7.0 torchvision torchaudio
pip install -r requirements.txt
```

**Note**: CPU mode will be significantly slower. CUDA 12.6+ is required for optimal SAM3 performance.

### 3. Create Test Image (Optional)

Place a test image at `app/test_image.jpg` for the `/test` endpoint. If not present, a placeholder will be created automatically.

## Running the Service

### Development Mode

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Production Mode

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The service will be available at `http://localhost:8000`

## API Endpoints

### GET `/`

Root endpoint returning service information.

**Response:**
```json
{
  "service": "LockerLink AI",
  "status": "running",
  "version": "1.0.0"
}
```

### GET `/test`

Test endpoint that runs SAM3 inference on a sample image.

**Response:**
```json
{
  "status": "ok",
  "num_masks": 1,
  "message": "SAM3 inference successful"
}
```

**Usage:**
```bash
curl http://localhost:8000/test
```

### GET `/health`

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

## Docker Deployment

### Prerequisites

1. Ensure SAM3 repository is cloned in `sam3/` directory
2. Ensure SAM3 weights are downloaded in `models/sam3_weights/`
3. Docker with NVIDIA Container Toolkit installed for CUDA support

### Build Docker Image

```bash
docker build -t lockerlink-ai:latest .
```

### Run Docker Container

```bash
# With GPU support
docker run --gpus all -p 8000:8000 lockerlink-ai:latest

# Or with specific GPU
docker run --gpus device=0 -p 8000:8000 lockerlink-ai:latest
```

### Docker Compose (Optional)

Create a `docker-compose.yml`:

```yaml
version: '3.8'

services:
  lockerlink-ai:
    build: .
    ports:
      - "8000:8000"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    volumes:
      - ./models/sam3_weights:/app/models/sam3_weights
```

## Usage Examples

### Testing the Service

```bash
# Test endpoint
curl http://localhost:8000/test

# Health check
curl http://localhost:8000/health
```

### Using the Python API

```python
from app.sam3_inference import load_sam3_image_model, segment_image
from app.video_utils import download_video, extract_frames
from PIL import Image

# Load model
model, processor = load_sam3_image_model()

# Segment an image
image = Image.open("path/to/image.jpg")
result = segment_image(model, processor, image, text_prompt="a volleyball player")

# Download and process video
video_path = download_video("https://example.com/video.mp4")
frames = extract_frames(video_path, fps=10)

# Process each frame
for frame in frames:
    result = segment_image(model, processor, frame, text_prompt="a volleyball player")
```

## CUDA Compatibility

- **Required**: CUDA 12.6+
- **PyTorch**: 2.7.0+ with CUDA support
- **GPU**: NVIDIA GPU with CUDA compute capability 7.0+

The service will automatically detect and use CUDA if available. If CUDA is not available, it will fall back to CPU (with significantly reduced performance).

## Troubleshooting

### SAM3 Import Errors

If you see import errors for SAM3:

```bash
# Ensure SAM3 is cloned
ls sam3/

# Reinstall SAM3
pip install -e ./sam3
```

### Checkpoint Not Found

If the model fails to load:

1. Verify checkpoints are in `models/sam3_weights/`
2. Check checkpoint filenames match expected patterns
3. Ensure you have access to the HuggingFace repository

### CUDA Errors

If CUDA is not detected:

1. Verify CUDA 12.6+ is installed: `nvcc --version`
2. Verify PyTorch CUDA support: `python -c "import torch; print(torch.cuda.is_available())"`
3. Check GPU is accessible: `nvidia-smi`

## Development

### Adding New Endpoints

Add new endpoints in `app/main.py` following the FastAPI pattern:

```python
@app.post("/analyze")
async def analyze_video(video_url: str):
    # Your analysis logic
    pass
```

### Extending SAM3 Inference

Modify `app/sam3_inference.py` to add custom segmentation functions or integrate with video processing pipelines.

## License

This project is part of the LockerLink ecosystem. SAM3 is licensed under Apache 2.0 (see SAM3 repository for details).

## References

- [SAM3 GitHub Repository](https://github.com/facebookresearch/sam3)
- [SAM3 HuggingFace](https://huggingface.co/facebook/sam3)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [PyTorch Documentation](https://pytorch.org/docs/)

