"""
Video download and frame extraction utilities.
Handles downloading videos from Firebase/Supabase and extracting frames with OpenCV.
"""

import os
import tempfile
import logging
import time
from pathlib import Path
from typing import List, Optional
import cv2
import requests
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)


def download_video(url: str, output_path: Optional[str] = None) -> str:
    """
    Download video from URL (Firebase/Supabase) to a temporary file.
    
    Args:
        url: URL to download video from
        output_path: Optional path to save video. If None, uses temp file.
    
    Returns:
        Path to downloaded video file
    """
    try:
        logger.info(f"Downloading video from: {url}")
        
        # Create output path if not provided
        if output_path is None:
            # Create temp file with .mp4 extension
            temp_dir = tempfile.gettempdir()
            temp_file = tempfile.NamedTemporaryFile(
                suffix='.mp4',
                dir=temp_dir,
                delete=False
            )
            output_path = temp_file.name
            temp_file.close()
        
        # Download video with streaming for large files
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        # Get file size if available
        total_size = int(response.headers.get('content-length', 0))
        if total_size > 0:
            logger.info(f"Video size: {total_size / (1024 * 1024):.2f} MB")
        
        # Write video to file
        downloaded = 0
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        if downloaded % (1024 * 1024) == 0:  # Log every MB
                            logger.debug(f"Download progress: {progress:.1f}%")
        
        logger.info(f"Video downloaded successfully to: {output_path}")
        return output_path
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download video: {e}")
        raise
    except Exception as e:
        logger.error(f"Error downloading video: {e}", exc_info=True)
        raise


def extract_frames(
    video_path: str,
    fps: int = 10,
    output_dir: Optional[str] = None,
    max_frames: Optional[int] = None
) -> List[Image.Image]:
    """
    Extract frames from video using OpenCV.
    
    Args:
        video_path: Path to video file
        fps: Target frames per second to extract (default: 10)
        output_dir: Optional directory to save frames as images.
                    If None, frames are returned as PIL Images only.
        max_frames: Optional maximum number of frames to extract
    
    Returns:
        List of PIL Images (frames)
    """
    try:
        logger.debug("=" * 80)
        logger.debug("EXTRACTING FRAMES FROM VIDEO")
        logger.debug(f"Video path: {video_path}")
        logger.debug(f"Target FPS: {fps}")
        logger.debug(f"Max frames: {max_frames}")
        logger.debug("=" * 80)
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        logger.info(f"Extracting frames from: {video_path} at {fps} fps")
        logger.debug(f"File exists: {os.path.exists(video_path)}")
        logger.debug(f"File size: {os.path.getsize(video_path) / (1024*1024):.2f} MB")
        
        # Open video
        logger.debug("Opening video with OpenCV...")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")
        logger.debug("Video opened successfully")
        
        # Get video properties
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / video_fps if video_fps > 0 else 0
        
        logger.info(f"Video properties: {total_frames} frames, {video_fps:.2f} fps, {duration:.2f}s duration")
        logger.debug(f"Video resolution: {width}x{height}")
        logger.debug(f"Video codec: {int(cap.get(cv2.CAP_PROP_FOURCC))}")
        
        # Calculate frame interval
        frame_interval = max(1, int(video_fps / fps)) if video_fps > 0 else 1
        logger.info(f"Extracting every {frame_interval} frame(s)")
        logger.debug(f"Frame interval calculation: {video_fps} fps / {fps} target = {frame_interval}")
        
        frames = []
        frame_count = 0
        extracted_count = 0
        
        # Create output directory if specified
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        logger.debug("Starting frame extraction loop...")
        extraction_start = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.debug(f"End of video reached at frame {frame_count}")
                break
            
            # Extract frame at specified interval
            if frame_count % frame_interval == 0:
                extract_frame_start = time.time()
                
                # Convert BGR to RGB for PIL
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                frames.append(pil_image)
                extracted_count += 1
                
                extract_time = time.time() - extract_frame_start
                if extracted_count % 10 == 0 or extracted_count <= 5:
                    logger.debug(f"Extracted frame {extracted_count} (video frame {frame_count}) in {extract_time:.3f}s")
                
                # Save frame if output directory specified
                if output_dir:
                    frame_filename = f"frame_{extracted_count:06d}.jpg"
                    frame_path = os.path.join(output_dir, frame_filename)
                    pil_image.save(frame_path, quality=95)
                
                # Check max_frames limit
                if max_frames and extracted_count >= max_frames:
                    logger.info(f"Reached max_frames limit: {max_frames}")
                    logger.debug(f"Stopping extraction at frame {frame_count}")
                    break
            
            frame_count += 1
        
        extraction_time = time.time() - extraction_start
        cap.release()
        
        logger.info(f"Extracted {len(frames)} frames from video in {extraction_time:.2f}s")
        logger.debug(f"Average time per frame: {extraction_time/len(frames) if frames else 0:.3f}s")
        logger.debug(f"Frame extraction rate: {len(frames)/extraction_time if extraction_time > 0 else 0:.1f} frames/sec")
        logger.debug("=" * 80)
        
        return frames
        
    except Exception as e:
        logger.error(f"Error extracting frames: {e}", exc_info=True)
        raise


def create_annotated_video(
    annotated_frames: List[Dict],
    output_path: str,
    fps: float = 10.0
) -> str:
    """
    Create a video from annotated frames showing the analysis.
    
    Args:
        annotated_frames: List of dicts with 'frame' and 'image' (PIL Image) keys
        output_path: Path to save the output video
        fps: Frames per second for output video
    
    Returns:
        Path to created video file
    """
    logger.info(f"Creating annotated video with {len(annotated_frames)} frames at {fps} fps")
    logger.debug(f"Output path: {output_path}")
    
    if not annotated_frames:
        raise ValueError("No annotated frames provided")
    
    # Get video dimensions from first frame
    first_frame = annotated_frames[0]["image"]
    width, height = first_frame.size
    logger.debug(f"Video dimensions: {width}x{height}")
    
    # Define codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        raise ValueError(f"Failed to create video writer for {output_path}")
    
    try:
        for af in sorted(annotated_frames, key=lambda x: x["frame"]):
            # Convert PIL Image to OpenCV format
            img = af["image"]
            # Convert RGB to BGR for OpenCV
            img_array = np.array(img)
            if len(img_array.shape) == 3:
                img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            else:
                img_bgr = img_array
            
            out.write(img_bgr)
            logger.debug(f"Wrote frame {af['frame']} to video")
        
        logger.info(f"Successfully created annotated video: {output_path}")
        return output_path
        
    finally:
        out.release()


def extract_frame_at_time(
    video_path: str,
    timestamp: float,
    output_path: Optional[str] = None
) -> Image.Image:
    """
    Extract a single frame at a specific timestamp.
    
    Args:
        video_path: Path to video file
        timestamp: Timestamp in seconds
        output_path: Optional path to save the frame
    
    Returns:
        PIL Image of the frame
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")
        
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_number = int(timestamp * video_fps)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        
        if not ret:
            raise ValueError(f"Failed to read frame at timestamp {timestamp}s")
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        if output_path:
            pil_image.save(output_path, quality=95)
        
        cap.release()
        return pil_image
        
    except Exception as e:
        logger.error(f"Error extracting frame at time {timestamp}s: {e}", exc_info=True)
        raise

