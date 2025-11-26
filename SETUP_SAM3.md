# SAM3 Setup Guide - After HuggingFace Access Granted

Now that you have access to SAM3 on HuggingFace, follow these steps to set it up.

**We're using HuggingFace Transformers to load SAM3**, which is simpler and handles authentication automatically.

## Step 1: Authenticate with HuggingFace

First, authenticate with HuggingFace CLI so the model can auto-download:

```bash
# Install huggingface_hub if not already installed
pip install huggingface_hub

# Login to HuggingFace (you'll need your access token)
huggingface-cli login
```

You can get your access token from: https://huggingface.co/settings/tokens

## Step 2: Install Dependencies

```bash
# Install all dependencies (includes transformers, accelerate, huggingface_hub)
pip install -r requirements.txt
```

This will install:
- PyTorch
- FastAPI, uvicorn
- transformers (for SAM3)
- accelerate (for model optimization)
- huggingface_hub (for authentication)
- Other dependencies

**Note**: You don't need to clone the SAM3 repository anymore! We're using the official HuggingFace Transformers integration.

## Step 3: Verify Installation

The model will automatically download from HuggingFace on first use (via transformers). To test:

```bash
# Start the service
uvicorn app.main:app --host 0.0.0.0 --port 8000

# In another terminal, test it
curl http://localhost:8000/test
```

The first time you run this, SAM3 will download the model weights automatically (this may take a few minutes).

## Step 5: (Optional) Download Weights Manually

If you prefer to download weights manually instead of auto-download:

### Option A: Using HuggingFace CLI

```bash
# Install huggingface-cli if needed
pip install huggingface_hub

# Download model files
huggingface-cli download facebook/sam3 --local-dir models/sam3_weights
```

### Option B: Using Python

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="facebook/sam3",
    local_dir="models/sam3_weights",
    token="your_hf_token_here"  # Or set HF_TOKEN environment variable
)
```

### Option C: Using Git LFS

```bash
# Install git-lfs if not installed
# Then clone the model repository
cd models/sam3_weights
git lfs install
git clone https://huggingface.co/facebook/sam3 .
```

## Step 6: Verify Model Files

After manual download, you should see files like:
- `sam3_image_model.pth` or similar
- `sam3_video_model.pth` or similar
- `config.json`
- Other model files

## Step 7: Update Code to Use Local Weights (Optional)

If you downloaded weights manually, you can update the code to use them:

```python
# In app/sam3_inference.py, you can now pass checkpoint paths:
model, processor = load_sam3_image_model(checkpoint_path="models/sam3_weights/sam3_image_model.pth")
```

However, the default behavior (auto-download) should work fine.

## Troubleshooting

### "Model not found" error
- Make sure you're logged in: `huggingface-cli whoami`
- Verify access: Visit https://huggingface.co/facebook/sam3
- Check your token has read access

### "CUDA out of memory" error
- Reduce batch size or use CPU mode
- Process videos in smaller chunks
- Use "quick" analysis type instead of "full"

### Slow download
- Model weights are large (~2-4 GB)
- First download may take 10-30 minutes depending on connection
- Weights are cached after first download

### Import errors
- Make sure transformers is installed: `pip install transformers accelerate huggingface_hub`
- Check Python version: `python --version` (needs 3.12+)
- Verify imports: `python -c "from transformers import Sam3Model, Sam3Processor; print('OK')"`

## Next Steps

Once SAM3 is set up:

1. Test the service: `curl http://localhost:8000/test`
2. Check health: `curl http://localhost:8000/health`
3. Start analyzing videos via the `/analyze/video` endpoint
4. Integrate with LockerLink using the instructions in `INTEGRATION.md`

## Model Information

- **Model Size**: ~0.9B parameters
- **Format**: Safetensors (F32)
- **Auto-download**: Yes (requires HuggingFace authentication)
- **Local caching**: Weights are cached in HuggingFace cache directory

## Environment Variables (Optional)

You can set these to customize behavior:

```bash
# HuggingFace cache directory
export HF_HOME=/path/to/huggingface/cache

# HuggingFace token (alternative to CLI login)
export HF_TOKEN=your_token_here
```

