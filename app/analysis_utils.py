"""
Volleyball video analysis utilities.
Calculates metrics like vertical jump, approach speed, kill accuracy, etc.
"""

import logging
from typing import List, Dict, Any, Optional
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def analyze_volleyball_video(
    frames: List[Image.Image],
    image_model: Any,
    image_processor: Any,
    video_model: Any,
    analysis_type: str = "full"
) -> Dict[str, Any]:
    """
    Analyze a volleyball video to extract player and ball information.
    
    Args:
        frames: List of PIL Images (video frames)
        image_model: SAM3 image model
        image_processor: SAM3 image processor
        video_model: SAM3 video model
        analysis_type: Type of analysis ("full", "quick", "player_only", "ball_only")
    
    Returns:
        Dictionary with analysis results including:
        - player_tracks: List of player positions over time
        - ball_tracks: List of ball positions over time
        - key_frames: Important frames (jump, contact, etc.)
        - player_detected: Boolean
        - ball_detected: Boolean
        - action_type: "kill", "block", or "other"
    """
    logger.info(f"Analyzing {len(frames)} frames with analysis_type={analysis_type}")
    
    player_tracks = []
    ball_tracks = []
    key_frames = []
    
    player_detected = False
    ball_detected = False
    
    # Analyze each frame
    for frame_idx, frame in enumerate(frames):
        frame_results = {}
        
        # Detect player
        if analysis_type in ["full", "quick", "player_only"]:
            try:
                player_result = segment_image(
                    image_model,
                    image_processor,
                    frame,
                    text_prompt="a volleyball player"
                )
                
                if player_result.get("masks"):
                    player_detected = True
                    # Extract player bounding box and center
                    player_box = player_result.get("boxes", [None])[0] if player_result.get("boxes") else None
                    if player_box:
                        # Calculate center point
                        center_x = (player_box[0] + player_box[2]) / 2
                        center_y = (player_box[1] + player_box[3]) / 2
                        player_tracks.append({
                            "frame": frame_idx,
                            "x": center_x,
                            "y": center_y,
                            "box": player_box,
                            "confidence": player_result.get("scores", [0.0])[0] if player_result.get("scores") else 0.0
                        })
                        frame_results["player"] = player_tracks[-1]
            except Exception as e:
                logger.warning(f"Error detecting player in frame {frame_idx}: {e}")
        
        # Detect ball
        if analysis_type in ["full", "quick", "ball_only"]:
            try:
                ball_result = segment_image(
                    image_model,
                    image_processor,
                    frame,
                    text_prompt="a volleyball ball"
                )
                
                if ball_result.get("masks"):
                    ball_detected = True
                    ball_box = ball_result.get("boxes", [None])[0] if ball_result.get("boxes") else None
                    if ball_box:
                        center_x = (ball_box[0] + ball_box[2]) / 2
                        center_y = (ball_box[1] + ball_box[3]) / 2
                        ball_tracks.append({
                            "frame": frame_idx,
                            "x": center_x,
                            "y": center_y,
                            "box": ball_box,
                            "confidence": ball_result.get("scores", [0.0])[0] if ball_result.get("scores") else 0.0
                        })
                        frame_results["ball"] = ball_tracks[-1]
            except Exception as e:
                logger.warning(f"Error detecting ball in frame {frame_idx}: {e}")
        
        # Identify key frames (jump, contact, etc.)
        if frame_results:
            key_frames.append({
                "frame": frame_idx,
                "timestamp": frame_idx / 10.0,  # Assuming 10 fps
                **frame_results
            })
    
    # Classify action type
    action_type = classify_action(player_tracks, ball_tracks)
    
    return {
        "player_tracks": player_tracks,
        "ball_tracks": ball_tracks,
        "key_frames": key_frames,
        "player_detected": player_detected,
        "ball_detected": ball_detected,
        "action_type": action_type,
        "total_frames": len(frames)
    }


def classify_action(player_tracks: List[Dict], ball_tracks: List[Dict]) -> str:
    """
    Classify the volleyball action as "kill", "block", or "other".
    
    This is a simplified classification - in production, you'd use more sophisticated
    analysis of player movement patterns, ball trajectory, and court position.
    """
    if not player_tracks or not ball_tracks:
        return "other"
    
    # Simple heuristic: if player moves upward significantly and ball is near player, likely a kill
    # If player is stationary and ball comes toward them, likely a block
    
    player_y_positions = [track["y"] for track in player_tracks]
    ball_y_positions = [track["y"] for track in ball_tracks]
    
    if len(player_y_positions) < 3 or len(ball_y_positions) < 3:
        return "other"
    
    # Check for upward player movement (jump)
    player_y_change = min(player_y_positions) - max(player_y_positions)  # Negative = upward movement
    ball_y_change = ball_y_positions[-1] - ball_y_positions[0]
    
    # If player jumps significantly and ball moves downward, likely a kill
    if player_y_change < -50 and ball_y_change > 0:
        return "kill"
    
    # If player doesn't move much and ball comes toward them, likely a block
    if abs(player_y_change) < 20 and ball_y_change < 0:
        return "block"
    
    return "other"


def calculate_metrics(analysis_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate volleyball performance metrics from analysis results.
    
    Returns:
        Dictionary with metrics:
        - vertical_jump: Maximum vertical jump height (inches)
        - approach_speed: Average approach speed (m/s)
        - ball_touch_detected: Boolean
        - kill_accuracy: Percentage (if kill detected)
        - max_reach_height: Maximum reach height (inches)
        - contact_point: Frame where ball contact occurred
    """
    player_tracks = analysis_result.get("player_tracks", [])
    ball_tracks = analysis_result.get("ball_tracks", [])
    action_type = analysis_result.get("action_type", "other")
    
    metrics = {
        "vertical_jump": None,
        "approach_speed": None,
        "ball_touch_detected": False,
        "kill_accuracy": None,
        "max_reach_height": None,
        "contact_point": None,
        "action_type": action_type
    }
    
    if not player_tracks:
        return metrics
    
    # Calculate vertical jump (difference between lowest and highest player Y position)
    player_y_positions = [track["y"] for track in player_tracks]
    if len(player_y_positions) > 1:
        min_y = min(player_y_positions)  # Highest point (lower Y = higher on screen)
        max_y = max(player_y_positions)  # Lowest point (higher Y = lower on screen)
        jump_height_pixels = max_y - min_y
        
        # Convert pixels to inches (rough estimate: 1 pixel ≈ 0.1 inches for typical video)
        # This is a placeholder - actual conversion requires camera calibration
        metrics["vertical_jump"] = round(jump_height_pixels * 0.1, 1)
        metrics["max_reach_height"] = round(jump_height_pixels * 0.1, 1)
    
    # Calculate approach speed (horizontal movement of player)
    if len(player_tracks) > 1:
        player_x_positions = [track["x"] for track in player_tracks]
        total_distance = sum(
            abs(player_x_positions[i] - player_x_positions[i-1])
            for i in range(1, len(player_x_positions))
        )
        
        # Estimate speed (pixels per frame, convert to m/s)
        # Placeholder conversion - requires actual calibration
        time_elapsed = len(player_tracks) / 10.0  # Assuming 10 fps
        if time_elapsed > 0:
            speed_pixels_per_sec = total_distance / time_elapsed
            metrics["approach_speed"] = round(speed_pixels_per_sec * 0.01, 2)  # Rough conversion
    
    # Detect ball touch (when ball and player are close)
    if player_tracks and ball_tracks:
        for player_track in player_tracks:
            for ball_track in ball_tracks:
                if abs(player_track["frame"] - ball_track["frame"]) <= 2:  # Within 2 frames
                    distance = np.sqrt(
                        (player_track["x"] - ball_track["x"])**2 +
                        (player_track["y"] - ball_track["y"])**2
                    )
                    if distance < 100:  # Threshold in pixels
                        metrics["ball_touch_detected"] = True
                        metrics["contact_point"] = player_track["frame"]
                        break
            if metrics["ball_touch_detected"]:
                break
    
    # Calculate kill accuracy (placeholder - would need more sophisticated analysis)
    if action_type == "kill" and metrics["ball_touch_detected"]:
        # This is a simplified calculation
        # In production, you'd analyze ball trajectory after contact
        metrics["kill_accuracy"] = 75.0  # Placeholder
    
    return metrics

