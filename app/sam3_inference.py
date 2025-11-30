"""
SAM3 inference using HuggingFace Transformers library.
Uses local model loading via transformers library.
"""

import os
import logging
import torch
from typing import Dict, Any, Optional
from PIL import Image

logger = logging.getLogger(__name__)

# Global model and processor cache
_sam3_model = None
_sam3_processor = None
_device = None


def get_hf_token() -> Optional[str]:
    """Get HuggingFace token from environment."""
    token = os.environ.get('HF_TOKEN')
    if not token:
        try:
            from dotenv import load_dotenv
            from pathlib import Path
            BASE_DIR = Path(__file__).resolve().parent.parent
            load_dotenv(dotenv_path=BASE_DIR / '.env.local')
            load_dotenv(dotenv_path=BASE_DIR / '.env')
            token = os.environ.get('HF_TOKEN')
        except ImportError:
            pass
    return token


def get_device():
    """Get the appropriate device (CUDA if available, else CPU)."""
    global _device
    if _device is None:
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {_device}")
    return _device


def load_sam3_image_model():
    """
    Load SAM3 model and processor for image segmentation.
    Uses caching to avoid reloading on every call.
    """
    global _sam3_model, _sam3_processor
    
    if _sam3_model is not None and _sam3_processor is not None:
        return _sam3_model, _sam3_processor
    
    try:
        from transformers import Sam3Model, Sam3Processor
        
        logger.info("Loading SAM3 model and processor...")
        device = get_device()
        
        # Set HF token for authentication
        token = get_hf_token()
        if token:
            os.environ['HF_TOKEN'] = token
        
        # Load model and processor
        model = Sam3Model.from_pretrained("facebook/sam3", token=token).to(device)
        processor = Sam3Processor.from_pretrained("facebook/sam3", token=token)
        
        model.eval()  # Set to evaluation mode
        
        # Cache for reuse
        _sam3_model = model
        _sam3_processor = processor
        
        logger.info(f"SAM3 model loaded successfully on {device}")
        return model, processor
        
    except Exception as e:
        logger.error(f"Failed to load SAM3 model: {e}", exc_info=True)
        raise Exception(f"Failed to load SAM3 model: {str(e)}")


def segment_image(
    image: Image.Image,
    text_prompt: str = "a volleyball player",
    point_prompts: Optional[list] = None,
    box_prompt: Optional[list] = None
) -> Dict[str, Any]:
    """
    Run SAM3 segmentation using HuggingFace Transformers.
    
    Args:
        image: PIL Image to segment
        text_prompt: Text prompt (e.g., "a volleyball player")
        point_prompts: Optional list of (x, y) point coordinates
        box_prompt: Optional bounding box [x1, y1, x2, y2] in xyxy format
    
    Returns:
        Dictionary with masks, boxes, scores
    """
    try:
        # Load model if not already loaded
        import time
        inference_start = time.time()
        print(f"DEBUG [sam3_inference]: Starting segmentation with prompt: '{text_prompt}'")
        print(f"DEBUG [sam3_inference]: Image size: {image.size}, mode: {image.mode}")
        
        model_load_start = time.time()
        model, processor = load_sam3_image_model()
        model_load_time = time.time() - model_load_start
        print(f"DEBUG [sam3_inference]: Model loading took {model_load_time:.3f}s")
        
        device = get_device()
        print(f"DEBUG [sam3_inference]: Using device: {device}")
        
        logger.debug(f"Segmenting image with text prompt: '{text_prompt}'")
        
        # Clear any cached tensors before processing
        cache_start = time.time()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"DEBUG [sam3_inference]: Cleared CUDA cache in {time.time() - cache_start:.3f}s")
        
        # Prepare inputs based on prompt type
        input_prep_start = time.time()
        print(f"DEBUG [sam3_inference]: Preparing inputs...")
        try:
            if box_prompt:
                # Bounding box input
                input_boxes = [[box_prompt]]  # [batch, num_boxes, 4] in xyxy format
                input_boxes_labels = [[1]]  # 1 = positive box
                inputs = processor(
                    images=image,
                    input_boxes=input_boxes,
                    input_boxes_labels=input_boxes_labels,
                    return_tensors="pt"
                ).to(device)
            elif point_prompts:
                # Point prompts
                input_points = [[[point_prompts]]]  # [batch, object, point, coord]
                input_labels = [[[1] * len(point_prompts)]]  # All positive
                inputs = processor(
                    images=image,
                    input_points=input_points,
                    input_labels=input_labels,
                    return_tensors="pt"
                ).to(device)
            else:
                # Text prompt (default)
                print(f"DEBUG [sam3_inference]: Using text prompt: '{text_prompt}'")
                inputs = processor(
                    images=image,
                    text=text_prompt,
                    return_tensors="pt"
                ).to(device)
                print(f"DEBUG [sam3_inference]: Inputs prepared, keys: {list(inputs.keys())}")
                if 'pixel_values' in inputs:
                    print(f"DEBUG [sam3_inference]: Pixel values shape: {inputs['pixel_values'].shape}")
        except Exception as e:
            input_prep_time = time.time() - input_prep_start
            print(f"DEBUG [sam3_inference]: Input preparation failed after {input_prep_time:.3f}s: {type(e).__name__}: {str(e)}")
            logger.error(f"Error preparing inputs: {e}", exc_info=True)
            raise Exception(f"Failed to prepare inputs: {str(e)}")
        
        input_prep_time = time.time() - input_prep_start
        print(f"DEBUG [sam3_inference]: Input preparation completed in {input_prep_time:.3f}s")
        
        # Run inference with memory management
        inference_run_start = time.time()
        print(f"DEBUG [sam3_inference]: Running model inference...")
        with torch.no_grad():
            try:
                outputs = model(**inputs)
                inference_run_time = time.time() - inference_run_start
                print(f"DEBUG [sam3_inference]: Model inference completed in {inference_run_time:.2f}s")
            except RuntimeError as e:
                error_str = str(e).lower()
                if "out of memory" in error_str or "cuda" in error_str:
                    # Clear cache and try again
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    logger.warning(f"Memory error detected, clearing cache: {e}")
                    # Try with smaller image or retry
                    try:
                        outputs = model(**inputs)
                    except RuntimeError as e2:
                        logger.error(f"Retry also failed: {e2}")
                        raise Exception(f"Out of memory error: {str(e2)}")
                else:
                    logger.error(f"Runtime error during inference: {e}", exc_info=True)
                    raise Exception(f"Model inference error: {str(e)}")
            except Exception as e:
                logger.error(f"Unexpected error during inference: {e}", exc_info=True)
                raise
        
        # Post-process results
        postprocess_start = time.time()
        print(f"DEBUG [sam3_inference]: Post-processing results...")
        try:
            results = processor.post_process_instance_segmentation(
                outputs,
                threshold=0.5,
                mask_threshold=0.5,
                target_sizes=inputs.get("original_sizes").tolist()
            )[0]
            
            num_objects = len(results["masks"]) if results.get("masks") is not None else 0
            postprocess_time = time.time() - postprocess_start
            print(f"DEBUG [sam3_inference]: Post-processing completed in {postprocess_time:.3f}s")
            print(f"DEBUG [sam3_inference]: Found {num_objects} objects")
            logger.debug(f"Found {num_objects} objects")
            
            # Clear intermediate tensors to free memory
            del outputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return {
                "masks": results["masks"],  # Tensor of masks
                "boxes": results["boxes"],   # Tensor of bounding boxes (xyxy format)
                "scores": results["scores"], # Tensor of confidence scores
            }
        except Exception as e:
            logger.error(f"Error post-processing results: {e}", exc_info=True)
            # Clear memory on error
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise Exception(f"Failed to post-process results: {str(e)}")
            total_inference_time = time.time() - inference_start
            print(f"DEBUG [sam3_inference]: Total segmentation time: {total_inference_time:.2f}s")
            print(f"DEBUG [sam3_inference]: Breakdown - Model load: {model_load_time:.2f}s, Input prep: {input_prep_time:.3f}s, Inference: {inference_run_time:.2f}s, Post-process: {postprocess_time:.3f}s")
        
    except Exception as e:
        total_inference_time = time.time() - inference_start if 'inference_start' in locals() else 0
        print(f"DEBUG [sam3_inference]: Segmentation failed after {total_inference_time:.2f}s: {type(e).__name__}: {str(e)}")
        logger.error(f"Error in segment_image: {e}", exc_info=True)
        raise Exception(f"Failed to segment image: {str(e)}")


def segment_video_frame(
    frame: Image.Image,
    text_prompt: str = "a volleyball player"
) -> Dict[str, Any]:
    """
    Run SAM3 inference on a single video frame.
    
    Args:
        frame: PIL Image frame to segment
        text_prompt: Optional text prompt
    
    Returns:
        Dictionary with masks, boxes, scores for the frame
    """
    # Video frames are just images - use the same function
    return segment_image(frame, text_prompt=text_prompt)


def load_sam3_video_model():
    """
    Load SAM3 video model (for future use if needed).
    Currently not implemented - using image model for frames.
    """
    logger.warning("load_sam3_video_model() - video model not yet implemented, using image model")
    return load_sam3_image_model()
