# Quick Start Guide - SAM3 Access Granted ✅

You now have access to SAM3! Follow these steps to get everything running.

## 🚀 Quick Setup (5 minutes)

### 1. Authenticate with HuggingFace

```bash
pip install huggingface_hub
huggingface-cli login
```

Enter your HuggingFace token when prompted (get it from https://huggingface.co/settings/tokens)

### 2. Install Dependencies (including Transformers)

```bash
# Install PyTorch with CUDA (if you have GPU)
pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Or CPU-only version
pip install torch==2.7.0 torchvision torchaudio

# Install other dependencies
pip install -r requirements.txt
```

### 3. Test It!

```bash
# Start the service
uvicorn app.main:app --host 0.0.0.0 --port 8000

# In another terminal, test it
curl http://localhost:8000/test
```

**First run will download the model (~2-4 GB) - this may take 10-30 minutes.**

## ✅ What Happens Next

1. **First Request**: Model auto-downloads from HuggingFace (one-time, ~10-30 min)
2. **Subsequent Requests**: Uses cached model (instant)
3. **Test Endpoint**: Creates a placeholder image and runs SAM3 on it
4. **Video Analysis**: Ready to analyze volleyball videos!

## 🎯 Next Steps

1. **Test the service**: `curl http://localhost:8000/test`
2. **Check health**: `curl http://localhost:8000/health`
3. **Read integration guide**: See `INTEGRATION.md` to connect with LockerLink
4. **Start analyzing videos**: Use the `/analyze/video` endpoint

## 📝 Important Notes

- **Model Auto-Download**: The model will automatically download from HuggingFace on first use
- **Authentication Required**: Make sure you're logged in with `huggingface-cli login`
- **CUDA Recommended**: GPU will make analysis much faster
- **First Download**: Be patient - the first model download takes time

## 🐛 Troubleshooting

### "Model not found" or authentication errors
```bash
# Verify you're logged in
huggingface-cli whoami

# Re-login if needed
huggingface-cli login
```

### Import errors
```bash
# Make sure transformers is installed
pip install transformers accelerate huggingface_hub

# Verify installation
python -c "from transformers import AutoImageProcessor, AutoModel; print('OK')"
```

### CUDA errors
- Check CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`
- If no GPU, the service will use CPU (slower but works)

### "ModuleNotFoundError" or "No module named 'app'"
```bash
# Make sure you're in the project root directory
cd "c:\Users\aarav\OneDrive\Documents\Projects\lockerlink ai"

# Verify the app directory exists
ls app/

# Try running with Python module syntax
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### "Address already in use" (port 8000 busy)
```bash
# Use a different port
uvicorn app.main:app --host 0.0.0.0 --port 8001

# Or find and kill the process using port 8000
# Windows:
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

### "ImportError" when starting uvicorn
```bash
# Make sure all dependencies are installed
pip install -r requirements.txt

# Verify imports work
python -c "from transformers import AutoImageProcessor, AutoModel; print('OK')"
python -c "import app.main; print('OK')"
```

## 📚 More Information

- **Full Setup Guide**: See `SETUP_SAM3.md`
- **Integration Guide**: See `INTEGRATION.md`
- **API Documentation**: See `README.md`

## 🎉 You're Ready!

Once the test endpoint works, you can:
- Analyze volleyball videos via `/analyze/video`
- Integrate with your LockerLink app
- Get metrics like vertical jump, approach speed, kill accuracy, etc.

Happy analyzing! 🏐

