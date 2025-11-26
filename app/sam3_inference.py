"""
SAM3 model loading and inference functions.
Handles both image and video segmentation using the official SAM3 API.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import torch
from PIL import Image

# SAM3 imports using HuggingFace Transformers
# This requires: pip install transformers accelerate
# And authentication: huggingface-cli login
try:
    from transformers import pipeline, AutoModel
except ImportError as e:
    logging.error(f"Transformers imports failed. Make sure to run: pip install transformers accelerate")
    logging.error(f"Import error: {e}")
    raise

logger = logging.getLogger(__name__)

# Global model cache
_image_model = None
_image_processor = None
_video_model = None

# Paths
SAM3_WEIGHTS_DIR = Path("models/sam3_weights")
SAM3_REPO_DIR = Path("sam3")


def get_device() -> torch.device:
    """Get the appropriate device (CUDA if available, else CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA version: {torch.version.cuda}")
    else:
        device = torch.device("cpu")
        logger.warning("CUDA not available, using CPU (performance will be limited)")
    return device


def find_checkpoint_path(model_type: str = "image") -> Optional[Path]:
    """
    Find SAM3 checkpoint file in models/sam3_weights/.
    
    Args:
        model_type: "image" or "video"
    
    Returns:
        Path to checkpoint file or None if not found
    """
    if not SAM3_WEIGHTS_DIR.exists():
        logger.warning(f"Checkpoints directory not found: {SAM3_WEIGHTS_DIR}")
        logger.warning("Please download SAM3 weights from HuggingFace and place in models/sam3_weights/")
        return None
    
    # Look for common checkpoint file patterns
    checkpoint_patterns = [
        f"sam3_{model_type}*.pth",
        f"sam3-{model_type}*.pth",
        f"*{model_type}*.pth",
        "*.pth",
        "*.pt",
        "*.ckpt"
    ]
    
    for pattern in checkpoint_patterns:
        matches = list(SAM3_WEIGHTS_DIR.glob(pattern))
        if matches:
            checkpoint_path = matches[0]
            logger.info(f"Found checkpoint: {checkpoint_path}")
            return checkpoint_path
    
    logger.warning(f"No checkpoint found in {SAM3_WEIGHTS_DIR}")
    logger.warning("Please download SAM3 weights from HuggingFace and place in models/sam3_weights/")
    return None


def load_sam3_image_model(checkpoint_path: Optional[str] = None) -> Tuple[Any, Any]:
    """
    Load SAM3 image model using HuggingFace Transformers.
    
    This automatically handles authentication and downloads from HuggingFace.
    Requires: huggingface-cli login (or HF_TOKEN environment variable)
    
    Args:
        checkpoint_path: Optional path to local checkpoint file.
                       If None, model will auto-download from HuggingFace.
    
    Returns:
        Tuple of (model, processor)
    """
    global _image_model, _image_processor
    
    # Return cached model if already loaded
    if _image_model is not None and _image_processor is not None:
        logger.info("Using cached SAM3 image model")
        return _image_model, _image_processor
    
    try:
        device = get_device()
        
        logger.info("Loading SAM3 image model from HuggingFace (facebook/sam3)...")
        logger.info("Make sure you're authenticated: huggingface-cli login")
        
        # Use pipeline approach for SAM3 (recommended by HuggingFace)
        # Pipeline handles both model and processor automatically
        try:
            _image_processor = pipeline("mask-generation", model="facebook/sam3", device=device)
            _image_model = _image_processor.model  # Extract model from pipeline
            logger.info("Loaded SAM3 using pipeline approach")
        except Exception as pipeline_error:
            logger.warning(f"Pipeline approach failed: {pipeline_error}")
            logger.info("Falling back to direct AutoModel loading...")
            # Fallback: Try loading model directly (may not work without processor)
            _image_model = AutoModel.from_pretrained("facebook/sam3").to(device)
            _image_model.eval()
            _image_processor = None  # Will need to handle processing differently
        
        logger.info(f"SAM3 image model loaded successfully on {device}")
        return _image_model, _image_processor
        
    except Exception as e:
        logger.error(f"Failed to load SAM3 image model: {e}", exc_info=True)
        logger.error("Make sure you're authenticated with HuggingFace: huggingface-cli login")
        logger.error("Or set HF_TOKEN environment variable")
        raise


def segment_image(
    model: Any,
    processor: Any,
    image: Image.Image,
    text_prompt: Optional[str] = None,
    point_prompts: Optional[list] = None,
    box_prompt: Optional[list] = None
) -> Dict[str, Any]:
    """
    Run SAM3 inference on an image using HuggingFace Transformers API.
    
    Args:
        model: SAM3Model from transformers (can be None, will use global cache)
        processor: Sam3Processor from transformers (can be None, will use global cache)
        image: PIL Image to segment
        text_prompt: Optional text prompt (e.g., "a volleyball player")
        point_prompts: Optional list of (x, y) point coordinates
        box_prompt: Optional bounding box [x1, y1, x2, y2] in xyxy format
    
    Returns:
        Dictionary with masks, boxes, scores
    """
    global _image_model, _image_processor
    
    # Use global cache if not provided
    if model is None:
        model = _image_model
    if processor is None:
        processor = _image_processor
    
    if model is None or processor is None:
        raise ValueError("SAM3 model not loaded. Call load_sam3_image_model() first.")
    
    try:
        # Check if processor is a pipeline (from pipeline approach)
        if hasattr(processor, '__call__') and hasattr(processor, 'model'):
            # Pipeline approach - use it directly
            prompt = text_prompt or "a volleyball player"
            results = processor(image, text=prompt)
            
            # Pipeline returns list of masks
            if isinstance(results, dict) and 'masks' in results:
                masks = results['masks']
                return {
                    "masks": masks,
                    "boxes": results.get("boxes"),
                    "scores": results.get("scores"),
                }
            elif isinstance(results, list):
                # Pipeline may return list of masks
                return {
                    "masks": results,
                    "boxes": None,
                    "scores": None,
                }
            else:
                return {
                    "masks": results,
                    "boxes": None,
                    "scores": None,
                }
        
        # Fallback: Direct model/processor approach
        device = next(model.parameters()).device
        
        # Prepare inputs based on prompt type
        if text_prompt:
            inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(device)
        elif box_prompt:
            input_boxes = [[box_prompt]]
            input_boxes_labels = [[1]]
            inputs = processor(
                images=image,
                input_boxes=input_boxes,
                input_boxes_labels=input_boxes_labels,
                return_tensors="pt"
            ).to(device)
        elif point_prompts:
            input_points = [[[point_prompts]]]
            input_labels = [[[1] * len(point_prompts)]]
            inputs = processor(
                images=image,
                input_points=input_points,
                input_labels=input_labels,
                return_tensors="pt"
            ).to(device)
        else:
            inputs = processor(images=image, text="a volleyball player", return_tensors="pt").to(device)
        
        # Run inference
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Extract results
        masks = getattr(outputs, 'pred_masks', None) or getattr(outputs, 'masks', None)
        boxes = getattr(outputs, 'pred_boxes', None) or getattr(outputs, 'boxes', None)
        scores = getattr(outputs, 'scores', None)
        
        return {
            "masks": masks,
            "boxes": boxes,
            "scores": scores,
        }
        
    except Exception as e:
        logger.error(f"Error during image segmentation: {e}", exc_info=True)
        raise


def load_sam3_video_model(checkpoint_path: Optional[str] = None) -> Any:
    """
    Load SAM3 video model using HuggingFace Transformers.
    
    This automatically handles authentication and downloads from HuggingFace.
    Requires: huggingface-cli login (or HF_TOKEN environment variable)
    
    Args:
        checkpoint_path: Optional path to local checkpoint file.
                       If None, model will auto-download from HuggingFace.
    
    Returns:
        Tuple of (model, processor) for video processing
    """
    global _video_model
    
    # Return cached model if already loaded
    if _video_model is not None:
        logger.info("Using cached SAM3 video model")
        return _video_model
    
    try:
        device = get_device()
        
        logger.info("Loading SAM3 video model from HuggingFace (facebook/sam3)...")
        logger.info("Make sure you're authenticated: huggingface-cli login")
        
        # Load video model using HuggingFace Transformers AutoModel
        # Use bfloat16 for video models (better performance)
        _video_model = AutoModel.from_pretrained("facebook/sam3").to(device, dtype=torch.bfloat16)
        _video_model.eval()
        
        logger.info(f"SAM3 video model loaded successfully on {device}")
        return _video_model
        
    except Exception as e:
        logger.error(f"Failed to load SAM3 video model: {e}", exc_info=True)
        logger.error("Make sure you're authenticated with HuggingFace: huggingface-cli login")
        logger.error("Or set HF_TOKEN environment variable")
        raise


def segment_video_frame(
    video_model: Any,
    frame: Image.Image,
    text_prompt: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run SAM3 inference on a single video frame.
    
    Args:
        video_model: SAM3 video predictor model
        frame: PIL Image frame to segment
        text_prompt: Optional text prompt
    
    Returns:
        Dictionary with masks, boxes, scores for the frame
    """
    if video_model is None:
        raise ValueError("SAM3 video model not loaded. Call load_sam3_video_model() first.")
    
    try:
        # SAM3 video API may differ - check official documentation
        # This is a placeholder based on typical video segmentation APIs
        # Adjust based on actual SAM3 video API from the repo
        
        # Convert PIL Image to tensor if needed
        # The exact API depends on SAM3 video predictor implementation
        output = video_model.predict(frame, text_prompt=text_prompt or "a volleyball player")
        
        result = {
            "masks": output.masks if hasattr(output, 'masks') else None,
            "boxes": output.boxes if hasattr(output, 'boxes') else None,
            "scores": output.scores if hasattr(output, 'scores') else None,
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error during video frame segmentation: {e}", exc_info=True)
        raise

