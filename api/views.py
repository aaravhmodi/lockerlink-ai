"""
Django views for LockerLink AI microservice.
Provides endpoints for SAM3-based volleyball video analysis.
"""

import os
import json
import logging
import time
import base64
from io import BytesIO
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.views.decorators.clickjacking import xframe_options_exempt
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from PIL import Image

from app.sam3_inference import load_sam3_image_model, load_sam3_video_model, segment_image
from app.video_utils import download_video, extract_frames
from app.analysis_utils import analyze_volleyball_video, calculate_metrics

logger = logging.getLogger(__name__)

# Global model cache
_sam3_model = None
_sam3_processor = None
_sam3_video_model = None


def _lazy_load_models():
    """Lazy load SAM3 models if not already loaded."""
    global _sam3_model, _sam3_processor, _sam3_video_model
    
    if _sam3_model is None or _sam3_processor is None:
        logger.info("Lazy loading SAM3 image model...")
        _sam3_model, _sam3_processor = load_sam3_image_model()
    
    if _sam3_video_model is None:
        logger.info("Lazy loading SAM3 video model...")
        _sam3_video_model = load_sam3_video_model()


@require_http_methods(["GET"])
def root(request):
    """Root endpoint - returns JSON for API calls."""
    return JsonResponse({
        "service": "LockerLink AI",
        "status": "running",
        "version": "1.0.0"
    })


def home_view(request):
    """Home page view - renders the frontend HTML."""
    from django.shortcuts import render
    return render(request, 'index.html')


@require_http_methods(["GET"])
def test_endpoint(request):
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
        output = segment_image(_sam3_model, _sam3_processor, image, text_prompt="a volleyball player")
        
        # Extract number of masks from output
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
        
        return JsonResponse({
            "status": "ok",
            "num_masks": num_masks,
            "message": "SAM3 inference successful"
        })
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return JsonResponse({"error": f"Test image not found: {e}"}, status=404)
    except Exception as e:
        logger.error(f"Error in test endpoint: {e}", exc_info=True)
        return JsonResponse({"error": f"Internal server error: {str(e)}"}, status=500)


@require_http_methods(["GET"])
def health_check(request):
    """Health check endpoint."""
    global _sam3_model, _sam3_video_model
    return JsonResponse({
        "status": "healthy",
        "model_loaded": _sam3_model is not None,
        "video_model_loaded": _sam3_video_model is not None
    })


@csrf_exempt
@require_http_methods(["POST"])
def analyze_video(request):
    """
    Analyze a volleyball highlight video.
    
    Accepts JSON with:
    - video_url: URL to video
    - highlight_id: Optional highlight ID
    - user_id: Optional user ID
    - analysis_type: Optional analysis type ("full", "quick", "player_only", "ball_only")
    
    Returns:
        JSON with analysis results and metrics
    """
    global _sam3_model, _sam3_processor, _sam3_video_model
    
    try:
        # Parse JSON request body
        data = json.loads(request.body)
        video_url = data.get('video_url')
        highlight_id = data.get('highlight_id')
        user_id = data.get('user_id')
        analysis_type = data.get('analysis_type', 'full')
        
        if not video_url:
            return JsonResponse({"error": "video_url is required"}, status=400)
        
        logger.info(f"Starting video analysis for: {video_url}")
        
        # Lazy load models if needed
        _lazy_load_models()
        
        # Download video
        logger.info("Downloading video...")
        video_path = download_video(video_url)
        
        try:
            # Extract frames for analysis
            logger.info("Extracting frames...")
            frames = extract_frames(video_path, fps=10, max_frames=100)
            
            if not frames:
                return JsonResponse({"error": "No frames extracted from video"}, status=400)
            
            logger.info(f"Extracted {len(frames)} frames for analysis")
            
            # Run analysis
            logger.info("Running volleyball video analysis...")
            analysis_result = analyze_volleyball_video(
                frames=frames,
                image_model=_sam3_model,
                image_processor=_sam3_processor,
                video_model=_sam3_video_model,
                analysis_type=analysis_type
            )
            
            # Calculate metrics
            metrics = calculate_metrics(analysis_result)
            
            logger.info("Analysis completed successfully")
            
            return JsonResponse({
                "status": "success",
                "highlight_id": highlight_id,
                "user_id": user_id,
                "analysis": {
                    "frames_analyzed": len(frames),
                    "player_detected": analysis_result.get("player_detected", False),
                    "ball_detected": analysis_result.get("ball_detected", False),
                    "action_type": analysis_result.get("action_type", "unknown"),
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
        
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON in request body"}, status=400)
    except Exception as e:
        logger.error(f"Error analyzing video: {e}", exc_info=True)
        return JsonResponse({"error": f"Video analysis failed: {str(e)}"}, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def analyze_batch_videos(request):
    """
    Analyze multiple videos in batch.
    
    Accepts JSON with list of video URLs.
    """
    try:
        data = json.loads(request.body)
        video_urls = data.get('video_urls', [])
        
        if not isinstance(video_urls, list):
            return JsonResponse({"error": "video_urls must be a list"}, status=400)
        
        results = []
        for video_url in video_urls:
            try:
                # Call analyze_video logic directly
                global _sam3_model, _sam3_processor, _sam3_video_model
                
                _lazy_load_models()
                
                video_path = download_video(video_url)
                try:
                    frames = extract_frames(video_path, fps=10, max_frames=100)
                    if not frames:
                        results.append({
                            "status": "error",
                            "video_url": video_url,
                            "error": "No frames extracted"
                        })
                        continue
                    
                    analysis_result = analyze_volleyball_video(
                        frames=frames,
                        image_model=_sam3_model,
                        image_processor=_sam3_processor,
                        video_model=_sam3_video_model,
                        analysis_type="full"
                    )
                    
                    metrics = calculate_metrics(analysis_result)
                    
                    results.append({
                        "status": "success",
                        "video_url": video_url,
                        "analysis": {
                            "frames_analyzed": len(frames),
                            "player_detected": analysis_result.get("player_detected", False),
                            "ball_detected": analysis_result.get("ball_detected", False),
                            "action_type": analysis_result.get("action_type", "unknown"),
                        },
                        "metrics": metrics
                    })
                finally:
                    if os.path.exists(video_path):
                        try:
                            os.remove(video_path)
                        except Exception:
                            pass
            except Exception as e:
                logger.error(f"Failed to analyze {video_url}: {e}")
                results.append({
                    "status": "error",
                    "video_url": video_url,
                    "error": str(e)
                })
        
        return JsonResponse({
            "status": "completed",
            "total": len(video_urls),
            "successful": sum(1 for r in results if r.get("status") == "success"),
            "results": results
        })
        
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON in request body"}, status=400)
    except Exception as e:
        logger.error(f"Error in batch analysis: {e}", exc_info=True)
        return JsonResponse({"error": f"Batch analysis failed: {str(e)}"}, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def upload_video(request):
    """
    Upload and analyze a video file.
    
    Accepts multipart/form-data with 'video' file.
    Returns analysis results with extensive debugging info.
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
        logger.debug("=" * 80)
        logger.debug("VIDEO UPLOAD REQUEST STARTED")
        logger.debug("=" * 80)
        
        if 'video' not in request.FILES:
            debug_info['errors'].append('No video file in request')
            return JsonResponse({"error": "No video file provided"}, status=400)
        
        video_file = request.FILES['video']
        analysis_type = request.POST.get('analysis_type', 'full')
        
        debug_info['steps'].append(f'File received: {video_file.name} ({video_file.size} bytes)')
        logger.debug(f"File: {video_file.name}, Size: {video_file.size} bytes, Type: {video_file.content_type}")
        
        # Save uploaded file temporarily
        debug_info['steps'].append('Saving uploaded file')
        logger.debug("Saving uploaded file to temporary location...")
        file_path = default_storage.save(f'temp/{video_file.name}', ContentFile(video_file.read()))
        full_path = default_storage.path(file_path)
        debug_info['file_path'] = full_path
        logger.debug(f"File saved to: {full_path}")
        
        try:
            # Lazy load models
            debug_info['steps'].append('Loading SAM3 models')
            logger.debug("Loading SAM3 models...")
            _lazy_load_models()
            debug_info['model_loaded'] = _sam3_model is not None
            debug_info['video_model_loaded'] = _sam3_video_model is not None
            logger.debug(f"Models loaded - Image: {_sam3_model is not None}, Video: {_sam3_video_model is not None}")
            
            # Extract frames
            debug_info['steps'].append('Extracting frames from video')
            logger.debug("Extracting frames from video...")
            extract_start = time.time()
            frames = extract_frames(full_path, fps=10, max_frames=100)
            extract_time = time.time() - extract_start
            debug_info['frames_extracted'] = len(frames)
            debug_info['extraction_time'] = f"{extract_time:.2f}s"
            logger.debug(f"Extracted {len(frames)} frames in {extract_time:.2f}s")
            
            if not frames:
                debug_info['errors'].append('No frames extracted')
                return JsonResponse({"error": "No frames extracted from video"}, status=400)
            
            # Run analysis
            debug_info['steps'].append('Running SAM3 analysis')
            logger.debug("=" * 80)
            logger.debug("STARTING SAM3 ANALYSIS")
            logger.debug(f"Frames to analyze: {len(frames)}")
            logger.debug(f"Analysis type: {analysis_type}")
            logger.debug("=" * 80)
            
            analysis_start = time.time()
            analysis_result = analyze_volleyball_video(
                frames=frames,
                image_model=_sam3_model,
                image_processor=_sam3_processor,
                video_model=_sam3_video_model,
                analysis_type=analysis_type
            )
            analysis_time = time.time() - analysis_start
            debug_info['analysis_time'] = f"{analysis_time:.2f}s"
            logger.debug(f"Analysis completed in {analysis_time:.2f}s")
            
            # Calculate metrics
            debug_info['steps'].append('Calculating metrics')
            logger.debug("Calculating performance metrics...")
            metrics = calculate_metrics(analysis_result)
            debug_info['metrics_calculated'] = True
            logger.debug(f"Metrics: {json.dumps(metrics, indent=2, default=str)}")
            
            # Prepare response with debug info
            total_time = time.time() - debug_info['start_time']
            debug_info['total_time'] = f"{total_time:.2f}s"
            debug_info['steps'].append('Analysis complete')
            
            logger.debug("=" * 80)
            logger.debug("ANALYSIS COMPLETE")
            logger.debug(f"Total time: {total_time:.2f}s")
            logger.debug("=" * 80)
            
            response = {
                "status": "success",
                "analysis": {
                    "frames_analyzed": len(frames),
                    "player_detected": analysis_result.get("player_detected", False),
                    "ball_detected": analysis_result.get("ball_detected", False),
                    "action_type": analysis_result.get("action_type", "unknown"),
                },
                "metrics": metrics,
                "raw_data": {
                    "player_tracks_count": len(analysis_result.get("player_tracks", [])),
                    "ball_tracks_count": len(analysis_result.get("ball_tracks", [])),
                    "key_frames_count": len(analysis_result.get("key_frames", [])),
                },
                "debug_info": debug_info
            }
            
            return JsonResponse(response)
            
        finally:
            # Clean up uploaded file
            try:
                if default_storage.exists(file_path):
                    default_storage.delete(file_path)
                    logger.debug(f"Cleaned up temporary file: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to clean up file: {e}")
                debug_info['warnings'].append(f"Failed to clean up file: {e}")
        
    except Exception as e:
        debug_info['errors'].append(str(e))
        logger.error(f"Error in upload_video: {e}", exc_info=True)
        debug_info['total_time'] = f"{time.time() - debug_info['start_time']:.2f}s"
        return JsonResponse({
            "error": f"Video analysis failed: {str(e)}",
            "debug_info": debug_info
        }, status=500)


@csrf_exempt
@require_http_methods(["GET", "POST"])
def test_swing_video(request):
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
        logger.debug("=" * 80)
        logger.debug("TEST SWING.MOV REQUEST")
        logger.debug("=" * 80)
        
        # Look for swing.mov in various locations
        possible_paths = [
            'swing.mov',
            'app/swing.mov',
            'media/swing.mov',
            'static/swing.mov',
            '../swing.mov',
        ]
        
        swing_path = None
        for path in possible_paths:
            if os.path.exists(path):
                swing_path = path
                break
        
        if not swing_path:
            debug_info['errors'].append('swing.mov not found in any expected location')
            debug_info['searched_paths'] = possible_paths
            logger.warning(f"swing.mov not found. Searched: {possible_paths}")
            return JsonResponse({
                "error": "swing.mov not found. Please place swing.mov in the project root or app/ directory.",
                "debug_info": debug_info
            }, status=404)
        
        debug_info['swing_path'] = swing_path
        debug_info['steps'].append(f'Found swing.mov at: {swing_path}')
        logger.debug(f"Found swing.mov at: {swing_path}")
        
        # Lazy load models
        debug_info['steps'].append('Loading SAM3 models')
        logger.debug("Loading SAM3 models...")
        _lazy_load_models()
        debug_info['model_loaded'] = _sam3_model is not None
        
        # Extract frames
        debug_info['steps'].append('Extracting frames')
        logger.debug("Extracting frames from swing.mov...")
        extract_start = time.time()
        frames = extract_frames(swing_path, fps=10, max_frames=100)
        extract_time = time.time() - extract_start
        debug_info['frames_extracted'] = len(frames)
        debug_info['extraction_time'] = f"{extract_time:.2f}s"
        logger.debug(f"Extracted {len(frames)} frames in {extract_time:.2f}s")
        
        if not frames:
            debug_info['errors'].append('No frames extracted')
            return JsonResponse({"error": "No frames extracted from swing.mov"}, status=400)
        
        # Run analysis
        debug_info['steps'].append('Running SAM3 analysis')
        logger.debug("=" * 80)
        logger.debug("ANALYZING SWING.MOV")
        logger.debug(f"Frames: {len(frames)}")
        logger.debug("=" * 80)
        
        analysis_start = time.time()
        analysis_result = analyze_volleyball_video(
            frames=frames,
            image_model=_sam3_model,
            image_processor=_sam3_processor,
            video_model=_sam3_video_model,
            analysis_type="full"
        )
        analysis_time = time.time() - analysis_start
        debug_info['analysis_time'] = f"{analysis_time:.2f}s"
        logger.debug(f"Analysis completed in {analysis_time:.2f}s")
        
        # Calculate metrics
        debug_info['steps'].append('Calculating metrics')
        logger.debug("Calculating metrics...")
        metrics = calculate_metrics(analysis_result)
        
        total_time = time.time() - debug_info['start_time']
        debug_info['total_time'] = f"{total_time:.2f}s"
        debug_info['steps'].append('Analysis complete')
        
        logger.debug("=" * 80)
        logger.debug("SWING.MOV ANALYSIS COMPLETE")
        logger.debug(f"Total time: {total_time:.2f}s")
        logger.debug("=" * 80)
        
        # Convert annotated frames to base64 for frontend display
        annotated_frames_data = []
        annotated_frames = analysis_result.get("annotated_frames", [])
        logger.debug(f"Converting {len(annotated_frames)} annotated frames to base64...")
        
        for af in annotated_frames[:20]:  # Limit to first 20 frames
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
                logger.debug(f"Converted frame {af['frame']} to base64")
            except Exception as e:
                logger.warning(f"Failed to encode frame {af.get('frame')}: {e}")
        
        logger.debug(f"Successfully converted {len(annotated_frames_data)} frames")
        
        response_data = {
            "status": "success",
            "video_file": "swing.mov",
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
            "raw_data": {
                "player_tracks_count": len(analysis_result.get("player_tracks", [])),
                "ball_tracks_count": len(analysis_result.get("ball_tracks", [])),
                "player_tracks": analysis_result.get("player_tracks", [])[:20],
                "ball_tracks": analysis_result.get("ball_tracks", [])[:20],
            },
            "debug_info": debug_info
        }
        
        response = JsonResponse(response_data)
        response["Access-Control-Allow-Origin"] = "*"
        response["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        response["Access-Control-Allow-Headers"] = "Content-Type"
        return response
        
    except Exception as e:
        debug_info['errors'].append(str(e))
        logger.error(f"Error in test_swing_video: {e}", exc_info=True)
        debug_info['total_time'] = f"{time.time() - debug_info['start_time']:.2f}s"
        return JsonResponse({
            "error": f"Swing analysis failed: {str(e)}",
            "debug_info": debug_info
        }, status=500)

