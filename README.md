# LockerLink AI - Volleyball Video Analysis

FastAPI microservice for analyzing volleyball highlight videos using SAM3 via HuggingFace Inference API.

## Features

- 🎥 Video analysis with player and ball tracking
- 📊 Metrics: vertical jump, hang time, touch point, action type
- 🎬 Annotated video output showing tracked objects
- 🚀 Serverless-ready (no GPU needed)
- ⚡ Fast inference via HuggingFace API

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set HuggingFace Token

Create `.env.local` file:

```bash
HF_TOKEN=your_huggingface_token_here
```

Get your token from: https://huggingface.co/settings/tokens

### 3. Run the Server

```bash
python run.py
```

Or with uvicorn:

```bash
uvicorn app.main:app --reload --port 8000
```

### 4. Test It

Open in browser:
- `http://localhost:8000/` - Service info
- `http://localhost:8000/docs` - Interactive API docs
- `http://localhost:8000/test` - Test SAM3

## API Endpoints

### `GET /`
Service information

### `GET /health`
Health check

### `GET /test`
Test SAM3 on sample image

### `POST /test/swing`
Analyze swing.mov file (if present)

### `POST /analyze/video`
Analyze video from URL

**Request:**
```json
{
  "video_url": "https://example.com/video.mp4",
  "highlight_id": "optional_id",
  "user_id": "optional_user_id",
  "analysis_type": "full"
}
```

### `POST /upload`
Upload and analyze video file

**Form data:**
- `video`: Video file
- `analysis_type`: "full" (optional)

### `POST /analyze/batch`
Analyze multiple videos

**Request:**
```json
{
  "video_urls": [
    "https://example.com/video1.mp4",
    "https://example.com/video2.mp4"
  ]
}
```

## Response Format

```json
{
  "status": "success",
  "analysis": {
    "frames_analyzed": 100,
    "player_detected": true,
    "ball_detected": true,
    "action_type": "kill",
    "jump_start_frame": 10,
    "jump_peak_frame": 15,
    "jump_end_frame": 20,
    "touch_frame": 16
  },
  "metrics": {
    "vertical_jump": 28.5,
    "hang_time": 0.5,
    "touch_point": {
      "frame": 16,
      "timestamp": 1.6
    },
    "ball_touch_detected": true,
    "action_type": "kill"
  },
  "annotated_video": "data:video/mp4;base64,...",
  "annotated_frames": [...],
  "debug_info": {...}
}
```

## Architecture

- **FastAPI** - Web framework
- **HuggingFace Inference API** - SAM3 model (no local models needed)
- **OpenCV** - Video processing
- **PIL** - Image handling

## Deployment

### Docker

```bash
docker build -t lockerlink-ai .
docker run -p 8000:8000 -e HF_TOKEN=your_token lockerlink-ai
```

### Serverless Platforms

Works on:
- Vercel
- Fly.io
- Render
- Railway
- Firebase Functions
- AWS Lambda

No GPU required - all inference via HuggingFace API!

## Environment Variables

- `HF_TOKEN` - HuggingFace API token (required)

## Project Structure

```
lockerlink-ai/
├── app/
│   ├── main.py              # FastAPI application
│   ├── sam3_inference.py    # HuggingFace API calls
│   ├── video_utils.py       # Video processing
│   └── analysis_utils.py    # Volleyball metrics
├── run.py                   # Server entry point
├── requirements.txt
├── Dockerfile
└── .env.local               # Your HF_TOKEN (not in git)
```

## Notes

- First API call may be slow (model loading on HF side)
- Subsequent calls are fast
- API is billed per request
- No local model downloads needed
