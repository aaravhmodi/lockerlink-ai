"""
FastAPI application for LockerLink AI microservice.
Provides endpoints for SAM3-based volleyball video analysis.
"""

import os
import json
import logging
import time
import base64
import tempfile
from io import BytesIO
from typing import Optional, List
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

from app.sam3_inference import segment_image
from app.video_utils import download_video, extract_frames, create_annotated_video
from app.analysis_utils import analyze_volleyball_video, calculate_metrics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="LockerLink AI",
    description="Volleyball video analysis using SAM3",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# No model loading needed - using HuggingFace Inference API


# Request/Response models
class AnalyzeVideoRequest(BaseModel):
    video_url: str
    highlight_id: Optional[str] = None
    user_id: Optional[str] = None
    analysis_type: str = "full"


class BatchAnalyzeRequest(BaseModel):
    video_urls: List[str]


# Root endpoint
@app.get("/")
def read_root():
    """Root endpoint - service info."""
    return {
        "service": "LockerLink AI",
        "status": "running",
        "version": "1.0.0"
    }


@app.get("/health")
def health_check():
    """Health check endpoint."""
    # Check if HF token is available
    hf_token = os.environ.get('HF_TOKEN')
    try:
        from dotenv import load_dotenv
        from pathlib import Path
        BASE_DIR = Path(__file__).resolve().parent.parent
        load_dotenv(dotenv_path=BASE_DIR / '.env.local')
        load_dotenv(dotenv_path=BASE_DIR / '.env')
        hf_token = hf_token or os.environ.get('HF_TOKEN')
    except:
        pass
    
    return {
        "status": "healthy",
        "api_mode": "HuggingFace Inference API",
        "token_configured": hf_token is not None
    }


@app.get("/test")
def test_endpoint():
    """
    Test endpoint that runs SAM3 inference on a sample image.
    
    Returns:
        JSON with status and number of masks detected
    """
    global _sam3_model, _sam3_processor
    
    try:
        # No model loading needed - using HuggingFace Inference API
        
        # Load test image
        test_image_path = "app/test_image.jpg"
        
        # Check if test image exists, if not create a placeholder
        if not os.path.exists(test_image_path):
            logger.warning(f"Test image not found at {test_image_path}, creating placeholder...")
            placeholder = Image.new('RGB', (640, 480), color='gray')
            placeholder.save(test_image_path)
            logger.info(f"Created placeholder image at {test_image_path}")
        
        # Load and process image
        image = Image.open(test_image_path)
        logger.info(f"Loaded test image: {image.size}")
        
        # Run SAM3 inference via HuggingFace API
        output = segment_image(image, text_prompt="a volleyball player")
        
        # Extract number of masks
        num_masks = 0
        if isinstance(output, dict):
            masks = output.get('masks')
            if masks is not None:
                if hasattr(masks, 'shape'):
                    num_masks = masks.shape[0] if len(masks.shape) > 0 else 1
                elif isinstance(masks, (list, tuple)):
                    num_masks = len(masks)
                else:
                    num_masks = 1
        
        logger.info(f"SAM3 inference completed, detected {num_masks} mask(s)")
        
        return {
            "status": "ok",
            "num_masks": num_masks,
            "message": "SAM3 inference successful"
        }
        
    except Exception as e:
        logger.error(f"Error in test endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Test failed: {str(e)}")


@app.post("/analyze/video")
async def analyze_video(request: AnalyzeVideoRequest):
    """
    Analyze a volleyball highlight video from URL.
    
    Accepts a video URL and returns analysis metrics.
    """
    global _sam3_model, _sam3_processor, _sam3_video_model
    
    debug_info = {
        'start_time': time.time(),
        'steps': [],
        'errors': [],
        'warnings': []
    }
    
    try:
        debug_info['steps'].append('Request received')
        logger.info(f"Analyzing video: {request.video_url}")
        
        # No model loading needed - using HuggingFace Inference API
        debug_info['api_mode'] = 'HuggingFace Inference API'
        
        # Download video
        debug_info['steps'].append('Downloading video')
        logger.info("Downloading video...")
        video_path = download_video(request.video_url)
        
        try:
            # Extract frames
            debug_info['steps'].append('Extracting frames')
            logger.info("Extracting frames...")
            frames = extract_frames(video_path, fps=10, max_frames=100)
            debug_info['frames_extracted'] = len(frames)
            
            if not frames:
                raise HTTPException(status_code=400, detail="No frames extracted from video")
            
            # Run analysis
            debug_info['steps'].append('Running SAM3 analysis via HuggingFace API')
            logger.info("Running volleyball video analysis via HuggingFace API...")
            analysis_result = analyze_volleyball_video(
                frames=frames,
                analysis_type=request.analysis_type
            )
            
            # Calculate metrics
            debug_info['steps'].append('Calculating metrics')
            metrics = calculate_metrics(analysis_result)
            
            # Create annotated video
            debug_info['steps'].append('Creating annotated video')
            annotated_frames = analysis_result.get("annotated_frames", [])
            video_base64 = None
            
            if annotated_frames:
                try:
                    video_fd, annotated_video_path = tempfile.mkstemp(suffix='.mp4', prefix='annotated_')
                    os.close(video_fd)
                    create_annotated_video(annotated_frames, annotated_video_path, fps=10.0)
                    
                    with open(annotated_video_path, 'rb') as video_file:
                        video_data = video_file.read()
                        video_base64 = base64.b64encode(video_data).decode()
                        debug_info['annotated_video_size'] = f"{len(video_data) / (1024*1024):.2f} MB"
                    
                    os.remove(annotated_video_path)
                except Exception as e:
                    logger.warning(f"Failed to create annotated video: {e}")
                    debug_info['warnings'].append(f"Failed to create annotated video: {e}")
            
            # Convert annotated frames to base64 (first 20 for preview)
            annotated_frames_data = []
            for af in annotated_frames[:20]:
                try:
                    img = af["image"]
                    buffered = BytesIO()
                    img.save(buffered, format="JPEG", quality=85)
                    img_str = base64.b64encode(buffered.getvalue()).decode()
                    annotated_frames_data.append({
                        "frame": af["frame"],
                        "image": f"data:image/jpeg;base64,{img_str}",
                        "has_player": af.get("has_player", False),
                        "has_ball": af.get("has_ball", False)
                    })
                except Exception as e:
                    logger.warning(f"Failed to encode frame: {e}")
            
            total_time = time.time() - debug_info['start_time']
            debug_info['total_time'] = f"{total_time:.2f}s"
            
            return {
                "status": "success",
                "highlight_id": request.highlight_id,
                "user_id": request.user_id,
                "analysis": {
                    "frames_analyzed": len(frames),
                    "player_detected": analysis_result.get("player_detected", False),
                    "ball_detected": analysis_result.get("ball_detected", False),
                    "action_type": analysis_result.get("action_type", "unknown"),
                    "jump_start_frame": analysis_result.get("jump_start_frame"),
                    "jump_peak_frame": analysis_result.get("jump_peak_frame"),
                    "jump_end_frame": analysis_result.get("jump_end_frame"),
                    "touch_frame": analysis_result.get("touch_frame"),
                },
                "metrics": metrics,
                "annotated_frames": annotated_frames_data,
                "annotated_video": f"data:video/mp4;base64,{video_base64}" if video_base64 else None,
                "raw_data": {
                    "player_tracks_count": len(analysis_result.get("player_tracks", [])),
                    "ball_tracks_count": len(analysis_result.get("ball_tracks", [])),
                    "player_tracks": analysis_result.get("player_tracks", [])[:20],
                    "ball_tracks": analysis_result.get("ball_tracks", [])[:20],
                },
                "debug_info": debug_info
            }
            
        finally:
            # Clean up downloaded video
            if os.path.exists(video_path):
                try:
                    os.remove(video_path)
                    logger.info(f"Cleaned up temporary video: {video_path}")
                except Exception as e:
                    logger.warning(f"Failed to clean up video file: {e}")
        
    except Exception as e:
        debug_info['errors'].append(str(e))
        logger.error(f"Error analyzing video: {e}", exc_info=True)
        debug_info['total_time'] = f"{time.time() - debug_info['start_time']:.2f}s"
        raise HTTPException(status_code=500, detail=f"Video analysis failed: {str(e)}")


@app.post("/upload")
async def upload_video(video: UploadFile = File(...), analysis_type: str = "full"):
    """
    Upload and analyze a video file.
    
    Accepts multipart/form-data with video file.
    """
    global _sam3_model, _sam3_processor, _sam3_video_model
    
    debug_info = {
        'start_time': time.time(),
        'steps': [],
        'errors': [],
        'warnings': []
    }
    
    try:
        debug_info['steps'].append('Request received')
        logger.info(f"Upload received: {video.filename} ({video.size} bytes)")
        
        # Save uploaded file temporarily
        debug_info['steps'].append('Saving uploaded file')
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            content = await video.read()
            tmp_file.write(content)
            video_path = tmp_file.name
        
        debug_info['file_path'] = video_path
        
        try:
            # No model loading needed - using HuggingFace Inference API
            debug_info['api_mode'] = 'HuggingFace Inference API'
            
            # Extract frames
            debug_info['steps'].append('Extracting frames')
            frames = extract_frames(video_path, fps=10, max_frames=100)
            debug_info['frames_extracted'] = len(frames)
            
            if not frames:
                raise HTTPException(status_code=400, detail="No frames extracted from video")
            
            # Run analysis
            debug_info['steps'].append('Running SAM3 analysis via HuggingFace API')
            analysis_result = analyze_volleyball_video(
                frames=frames,
                analysis_type=analysis_type
            )
            
            # Calculate metrics
            debug_info['steps'].append('Calculating metrics')
            metrics = calculate_metrics(analysis_result)
            
            # Create annotated video
            annotated_frames = analysis_result.get("annotated_frames", [])
            video_base64 = None
            
            if annotated_frames:
                try:
                    video_fd, annotated_video_path = tempfile.mkstemp(suffix='.mp4', prefix='annotated_')
                    os.close(video_fd)
                    create_annotated_video(annotated_frames, annotated_video_path, fps=10.0)
                    
                    with open(annotated_video_path, 'rb') as video_file:
                        video_data = video_file.read()
                        video_base64 = base64.b64encode(video_data).decode()
                    
                    os.remove(annotated_video_path)
                except Exception as e:
                    logger.warning(f"Failed to create annotated video: {e}")
            
            # Convert frames to base64
            annotated_frames_data = []
            for af in annotated_frames[:20]:
                try:
                    img = af["image"]
                    buffered = BytesIO()
                    img.save(buffered, format="JPEG", quality=85)
                    img_str = base64.b64encode(buffered.getvalue()).decode()
                    annotated_frames_data.append({
                        "frame": af["frame"],
                        "image": f"data:image/jpeg;base64,{img_str}",
                        "has_player": af.get("has_player", False),
                        "has_ball": af.get("has_ball", False)
                    })
                except Exception as e:
                    logger.warning(f"Failed to encode frame: {e}")
            
            total_time = time.time() - debug_info['start_time']
            debug_info['total_time'] = f"{total_time:.2f}s"
            
            return {
                "status": "success",
                "analysis": {
                    "frames_analyzed": len(frames),
                    "player_detected": analysis_result.get("player_detected", False),
                    "ball_detected": analysis_result.get("ball_detected", False),
                    "action_type": analysis_result.get("action_type", "unknown"),
                },
                "metrics": metrics,
                "annotated_frames": annotated_frames_data,
                "annotated_video": f"data:video/mp4;base64,{video_base64}" if video_base64 else None,
                "debug_info": debug_info
            }
            
        finally:
            if os.path.exists(video_path):
                try:
                    os.remove(video_path)
                except Exception as e:
                    logger.warning(f"Failed to clean up file: {e}")
        
    except Exception as e:
        debug_info['errors'].append(str(e))
        logger.error(f"Error in upload_video: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.post("/test/swing")
async def test_swing_video():
    """
    Test endpoint that analyzes swing.mov if it exists.
    """
    global _sam3_model, _sam3_processor, _sam3_video_model
    
    debug_info = {
        'start_time': time.time(),
        'steps': [],
        'errors': [],
        'warnings': []
    }
    
    try:
        debug_info['steps'].append('Test swing.mov request received')
        
        # Look for swing.mov
        possible_paths = ['swing.mov', 'app/swing.mov', 'media/swing.mov']
        swing_path = None
        
        for path in possible_paths:
            if os.path.exists(path):
                swing_path = path
                break
        
        if not swing_path:
            raise HTTPException(
                status_code=404,
                detail="swing.mov not found. Please place swing.mov in the project root or app/ directory."
            )
        
        debug_info['swing_path'] = swing_path
        # No model loading needed - using HuggingFace Inference API
        
        # Extract frames
        debug_info['steps'].append('Extracting frames')
        frames = extract_frames(swing_path, fps=10, max_frames=100)
        debug_info['frames_extracted'] = len(frames)
        
        if not frames:
            raise HTTPException(status_code=400, detail="No frames extracted from swing.mov")
        
        # Run analysis
        debug_info['steps'].append('Running SAM3 analysis via HuggingFace API')
        analysis_result = analyze_volleyball_video(
            frames=frames,
            analysis_type="full"
        )
        
        # Calculate metrics
        debug_info['steps'].append('Calculating metrics')
        metrics = calculate_metrics(analysis_result)
        
        # Create annotated video
        annotated_frames = analysis_result.get("annotated_frames", [])
        video_base64 = None
        
        if annotated_frames:
            try:
                video_fd, annotated_video_path = tempfile.mkstemp(suffix='.mp4', prefix='annotated_')
                os.close(video_fd)
                create_annotated_video(annotated_frames, annotated_video_path, fps=10.0)
                
                with open(annotated_video_path, 'rb') as video_file:
                    video_data = video_file.read()
                    video_base64 = base64.b64encode(video_data).decode()
                
                os.remove(annotated_video_path)
            except Exception as e:
                logger.warning(f"Failed to create annotated video: {e}")
        
        # Convert frames to base64
        annotated_frames_data = []
        for af in annotated_frames[:20]:
            try:
                img = af["image"]
                buffered = BytesIO()
                img.save(buffered, format="JPEG", quality=85)
                img_str = base64.b64encode(buffered.getvalue()).decode()
                annotated_frames_data.append({
                    "frame": af["frame"],
                    "image": f"data:image/jpeg;base64,{img_str}",
                    "has_player": af.get("has_player", False),
                    "has_ball": af.get("has_ball", False)
                })
            except Exception as e:
                logger.warning(f"Failed to encode frame: {e}")
        
        total_time = time.time() - debug_info['start_time']
        debug_info['total_time'] = f"{total_time:.2f}s"
        
        return {
            "status": "success",
            "video_file": "swing.mov",
            "analysis": {
                "frames_analyzed": len(frames),
                "player_detected": analysis_result.get("player_detected", False),
                "ball_detected": analysis_result.get("ball_detected", False),
                "action_type": analysis_result.get("action_type", "unknown"),
            },
            "metrics": metrics,
            "annotated_frames": annotated_frames_data,
            "annotated_video": f"data:video/mp4;base64,{video_base64}" if video_base64 else None,
            "debug_info": debug_info
        }
        
    except HTTPException:
        raise
    except Exception as e:
        debug_info['errors'].append(str(e))
        logger.error(f"Error in test_swing_video: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Swing analysis failed: {str(e)}")


@app.post("/analyze/batch")
async def analyze_batch_videos(request: BatchAnalyzeRequest):
    """
    Analyze multiple videos in batch.
    """
    results = []
    
    for video_url in request.video_urls:
        try:
            analyze_request = AnalyzeVideoRequest(video_url=video_url)
            result = await analyze_video(analyze_request)
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to analyze {video_url}: {e}")
            results.append({
                "status": "error",
                "video_url": video_url,
                "error": str(e)
            })
    
    return {
        "status": "completed",
        "total": len(request.video_urls),
        "successful": sum(1 for r in results if r.get("status") == "success"),
        "results": results
    }
