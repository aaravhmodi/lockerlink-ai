"""
FastAPI entry point for LockerLink AI microservice.
Provides endpoints for SAM3-based volleyball video analysis.
"""

import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List
from PIL import Image
import logging

from app.sam3_inference import load_sam3_image_model, load_sam3_video_model, segment_image, segment_video_frame
from app.video_utils import download_video, extract_frames
from app.analysis_utils import analyze_volleyball_video, calculate_metrics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="LockerLink AI",
    description="Volleyball video analysis using SAM3 for player and ball segmentation",
    version="1.0.0"
)

# Global model cache
_sam3_model = None
_sam3_processor = None
_sam3_video_model = None

# Request models
class VideoAnalysisRequest(BaseModel):
    video_url: str
    highlight_id: Optional[str] = None
    user_id: Optional[str] = None
    analysis_type: Optional[str] = "full"  # "full", "quick", "player_only", "ball_only"


@app.on_event("startup")
async def startup_event():
    """Load SAM3 model on startup."""
    global _sam3_model, _sam3_processor
    try:
        logger.info("Loading SAM3 image model on startup...")
        _sam3_model, _sam3_processor = load_sam3_image_model()
        logger.info("SAM3 model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load SAM3 model: {e}")
        # Don't fail startup, allow lazy loading on first request


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "LockerLink AI",
        "status": "running",
        "version": "1.0.0"
    }


@app.get("/test")
async def test_endpoint():
    """
    Test endpoint that runs SAM3 inference on a sample image.
    
    Returns:
        JSON with status and number of masks detected
    """
    global _sam3_model, _sam3_processor
    
    try:
        # Lazy load model if not already loaded
        if _sam3_model is None or _sam3_processor is None:
            logger.info("Lazy loading SAM3 model...")
            _sam3_model, _sam3_processor = load_sam3_image_model()
        
        # Load test image
        test_image_path = "app/test_image.jpg"
        
        # Check if test image exists, if not create a placeholder
        if not os.path.exists(test_image_path):
            logger.warning(f"Test image not found at {test_image_path}, creating placeholder...")
            # Create a simple placeholder image
            placeholder = Image.new('RGB', (640, 480), color='gray')
            placeholder.save(test_image_path)
            logger.info(f"Created placeholder image at {test_image_path}")
        
        # Load and process image
        image = Image.open(test_image_path)
        logger.info(f"Loaded test image: {image.size}")
        
        # Run SAM3 inference with a volleyball-related prompt
        # Using HuggingFace Transformers API
        from app.sam3_inference import segment_image
        output = segment_image(_sam3_model, _sam3_processor, image, text_prompt="a volleyball player")
        
        # Extract number of masks from output
        # SAM3 Transformers output structure: dict with "masks", "boxes", "scores" keys
        num_masks = 0
        if isinstance(output, dict):
            masks = output.get('masks')
            if masks is not None:
                # Masks from transformers are torch tensors
                if hasattr(masks, 'shape'):
                    num_masks = masks.shape[0] if len(masks.shape) > 0 else 1
                elif isinstance(masks, (list, tuple)):
                    num_masks = len(masks)
                else:
                    num_masks = 1
        
        logger.info(f"SAM3 inference completed, detected {num_masks} mask(s)")
        
        return JSONResponse({
            "status": "ok",
            "num_masks": num_masks,
            "message": "SAM3 inference successful"
        })
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise HTTPException(status_code=404, detail=f"Test image not found: {e}")
    except Exception as e:
        logger.error(f"Error in test endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": _sam3_model is not None,
        "video_model_loaded": _sam3_video_model is not None
    }


@app.post("/analyze/video")
async def analyze_video(request: VideoAnalysisRequest):
    """
    Analyze a volleyball highlight video.
    
    Accepts a video URL (from Cloudinary/Firebase) and returns analysis metrics:
    - Player segmentation and tracking
    - Ball detection and tracking
    - Vertical jump height
    - Approach speed
    - Ball touch detection
    - Kill/block classification
    - Kill accuracy
    
    Args:
        request: VideoAnalysisRequest with video_url and optional metadata
    
    Returns:
        JSON with analysis results and metrics
    """
    global _sam3_model, _sam3_processor, _sam3_video_model
    
    try:
        logger.info(f"Starting video analysis for: {request.video_url}")
        
        # Lazy load models if needed
        if _sam3_model is None or _sam3_processor is None:
            logger.info("Loading SAM3 image model...")
            _sam3_model, _sam3_processor = load_sam3_image_model()
        
        if _sam3_video_model is None:
            logger.info("Loading SAM3 video model...")
            _sam3_video_model = load_sam3_video_model()
        
        # Download video
        logger.info("Downloading video...")
        video_path = download_video(request.video_url)
        
        try:
            # Extract frames for analysis
            logger.info("Extracting frames...")
            frames = extract_frames(video_path, fps=10, max_frames=100)  # Limit to 100 frames for performance
            
            if not frames:
                raise HTTPException(status_code=400, detail="No frames extracted from video")
            
            logger.info(f"Extracted {len(frames)} frames for analysis")
            
            # Run analysis
            logger.info("Running volleyball video analysis...")
            analysis_result = analyze_volleyball_video(
                frames=frames,
                image_model=_sam3_model,
                image_processor=_sam3_processor,
                video_model=_sam3_video_model,
                analysis_type=request.analysis_type
            )
            
            # Calculate metrics
            metrics = calculate_metrics(analysis_result)
            
            logger.info("Analysis completed successfully")
            
            return JSONResponse({
                "status": "success",
                "highlight_id": request.highlight_id,
                "user_id": request.user_id,
                "analysis": {
                    "frames_analyzed": len(frames),
                    "player_detected": analysis_result.get("player_detected", False),
                    "ball_detected": analysis_result.get("ball_detected", False),
                    "action_type": analysis_result.get("action_type", "unknown"),  # "kill", "block", "other"
                },
                "metrics": metrics,
                "raw_data": {
                    "player_tracks": analysis_result.get("player_tracks", []),
                    "ball_tracks": analysis_result.get("ball_tracks", []),
                    "key_frames": analysis_result.get("key_frames", [])
                }
            })
            
        finally:
            # Clean up downloaded video
            if os.path.exists(video_path):
                try:
                    os.remove(video_path)
                    logger.info(f"Cleaned up temporary video: {video_path}")
                except Exception as e:
                    logger.warning(f"Failed to clean up video file: {e}")
        
    except Exception as e:
        logger.error(f"Error analyzing video: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Video analysis failed: {str(e)}")


@app.post("/analyze/batch")
async def analyze_batch_videos(video_urls: List[str]):
    """
    Analyze multiple videos in batch (for background processing).
    
    Args:
        video_urls: List of video URLs to analyze
    
    Returns:
        List of analysis results
    """
    results = []
    for video_url in video_urls:
        try:
            request = VideoAnalysisRequest(video_url=video_url)
            result = await analyze_video(request)
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to analyze {video_url}: {e}")
            results.append({
                "status": "error",
                "video_url": video_url,
                "error": str(e)
            })
    
    return JSONResponse({
        "status": "completed",
        "total": len(video_urls),
        "successful": sum(1 for r in results if r.get("status") == "success"),
        "results": results
    })

