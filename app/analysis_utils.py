"""
Volleyball video analysis utilities.
Calculates metrics like vertical jump, approach speed, kill accuracy, etc.
Now focuses on tracking the player closest to/interacting with the ball.
"""

import logging
import time
import json
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from app.sam3_inference import segment_image

logger = logging.getLogger(__name__)

# Console logging function for frontend
def console_log(message: str, level: str = "info"):
    """Log to console with timestamp and level."""
    try:
        # Try to get microseconds
        import datetime
        now = datetime.datetime.now()
        timestamp = now.strftime("%H:%M:%S") + ".{:03d}".format(now.microsecond // 1000)
    except:
        timestamp = time.strftime("%H:%M:%S")
    
    # Use % formatting to avoid f-string issues
    print("[%s] [%s] %s" % (timestamp, level.upper(), str(message)), flush=True)
    logger.debug("CONSOLE: %s", message)


def find_player_closest_to_ball(
    player_tracks: List[Dict],
    ball_tracks: List[Dict],
    frame_idx: int
) -> Optional[Dict]:
    """
    Find the player track closest to the ball at a given frame.
    
    Returns:
        Player track dict or None
    """
    console_log(f"Finding player closest to ball at frame {frame_idx}")
    
    if not player_tracks or not ball_tracks:
        console_log(f"No player or ball tracks available at frame {frame_idx}", "warning")
        return None
    
    # Find ball position at this frame (or closest frame)
    ball_at_frame = None
    min_frame_diff = float('inf')
    
    for ball_track in ball_tracks:
        frame_diff = abs(ball_track["frame"] - frame_idx)
        if frame_diff < min_frame_diff:
            min_frame_diff = frame_diff
            ball_at_frame = ball_track
    
    if not ball_at_frame:
        console_log(f"No ball found near frame {frame_idx}", "warning")
        return None
    
    ball_x, ball_y = ball_at_frame["x"], ball_at_frame["y"]
    console_log(f"Ball position at frame {frame_idx}: ({ball_x:.1f}, {ball_y:.1f})")
    
    # Find player closest to ball at this frame
    closest_player = None
    min_distance = float('inf')
    
    for player_track in player_tracks:
        frame_diff = abs(player_track["frame"] - frame_idx)
        if frame_diff <= 3:  # Within 3 frames
            distance = np.sqrt(
                (player_track["x"] - ball_x)**2 +
                (player_track["y"] - ball_y)**2
            )
            console_log(f"Player at frame {player_track['frame']}: distance to ball = {distance:.1f}px")
            
            if distance < min_distance:
                min_distance = distance
                closest_player = player_track
    
    if closest_player:
        console_log(f"Closest player found at frame {closest_player['frame']}, distance: {min_distance:.1f}px", "success")
    else:
        console_log(f"No player found close to ball at frame {frame_idx}", "warning")
    
    return closest_player


def analyze_volleyball_video(
    frames: List[Image.Image],
    image_model: Any,
    image_processor: Any,
    video_model: Any,
    analysis_type: str = "full"
) -> Dict[str, Any]:
    """
    Analyze a volleyball video to extract player and ball information.
    Focuses on tracking the player closest to/interacting with the ball.
    
    Args:
        frames: List of PIL Images (video frames)
        image_model: SAM3 image model
        image_processor: SAM3 image processor
        video_model: SAM3 video model
        analysis_type: Type of analysis ("full", "quick", "player_only", "ball_only")
    
    Returns:
        Dictionary with analysis results including:
        - player_tracks: List of player positions over time (filtered to closest player)
        - ball_tracks: List of ball positions over time
        - key_frames: Important frames (jump, contact, etc.)
        - player_detected: Boolean
        - ball_detected: Boolean
        - action_type: "kill", "block", or "other"
        - annotated_frames: List of frames with visual annotations
    """
    console_log("=" * 80)
    console_log("STARTING VOLLEYBALL VIDEO ANALYSIS")
    console_log(f"Total frames: {len(frames)}")
    console_log(f"Frame dimensions: {frames[0].size if frames else 'N/A'}")
    console_log(f"Analysis type: {analysis_type}")
    console_log("=" * 80)
    
    all_player_tracks = []  # All detected players
    ball_tracks = []
    key_frames = []
    annotated_frames = []  # Frames with visual annotations
    
    player_detected = False
    ball_detected = False
    
    # Step 1: Detect all players and ball in all frames
    console_log("STEP 1: Detecting players and ball in all frames...")
    for frame_idx, frame in enumerate(frames):
        console_log(f"\n--- Processing Frame {frame_idx + 1}/{len(frames)} ---")
        frame_results = {}
        
        # Detect all players
        if analysis_type in ["full", "quick", "player_only"]:
            try:
                console_log(f"Frame {frame_idx}: Detecting players...")
                detection_start = time.time()
                player_result = segment_image(
                    image_model,
                    image_processor,
                    frame,
                    text_prompt="a volleyball player"
                )
                detection_time = time.time() - detection_start
                console_log(f"Frame {frame_idx}: Player detection took {detection_time:.3f}s")
                
                if player_result.get("masks"):
                    player_detected = True
                    player_box = player_result.get("boxes", [None])[0] if player_result.get("boxes") else None
                    if player_box:
                        center_x = (player_box[0] + player_box[2]) / 2
                        center_y = (player_box[1] + player_box[3]) / 2
                        confidence = player_result.get("scores", [0.0])[0] if player_result.get("scores") else 0.0
                        
                        track = {
                            "frame": frame_idx,
                            "x": center_x,
                            "y": center_y,
                            "box": player_box,
                            "confidence": confidence,
                            "mask": player_result.get("masks")
                        }
                        all_player_tracks.append(track)
                        console_log(f"Frame {frame_idx}: Player detected at ({center_x:.1f}, {center_y:.1f}), confidence: {confidence:.3f}")
            except Exception as e:
                console_log(f"Frame {frame_idx}: Error detecting player: {e}", "error")
                logger.warning(f"Error detecting player in frame {frame_idx}: {e}", exc_info=True)
        
        # Detect ball
        if analysis_type in ["full", "quick", "ball_only"]:
            try:
                console_log(f"Frame {frame_idx}: Detecting ball...")
                detection_start = time.time()
                ball_result = segment_image(
                    image_model,
                    image_processor,
                    frame,
                    text_prompt="a volleyball ball"
                )
                detection_time = time.time() - detection_start
                console_log(f"Frame {frame_idx}: Ball detection took {detection_time:.3f}s")
                
                if ball_result.get("masks"):
                    ball_detected = True
                    ball_box = ball_result.get("boxes", [None])[0] if ball_result.get("boxes") else None
                    if ball_box:
                        center_x = (ball_box[0] + ball_box[2]) / 2
                        center_y = (ball_box[1] + ball_box[3]) / 2
                        confidence = ball_result.get("scores", [0.0])[0] if ball_result.get("scores") else 0.0
                        
                        track = {
                            "frame": frame_idx,
                            "x": center_x,
                            "y": center_y,
                            "box": ball_box,
                            "confidence": confidence,
                            "mask": ball_result.get("masks")
                        }
                        ball_tracks.append(track)
                        console_log(f"Frame {frame_idx}: Ball detected at ({center_x:.1f}, {center_y:.1f}), confidence: {confidence:.3f}")
            except Exception as e:
                console_log(f"Frame {frame_idx}: Error detecting ball: {e}", "error")
                logger.warning(f"Error detecting ball in frame {frame_idx}: {e}", exc_info=True)
    
    console_log(f"\nDetection complete: {len(all_player_tracks)} player detections, {len(ball_tracks)} ball detections")
    
    # Step 2: Find the player closest to ball during jump/interaction
    console_log("\nSTEP 2: Identifying player closest to ball...")
    interaction_frames = []
    
    # Find frames where ball and player are close (potential interaction)
    for ball_track in ball_tracks:
        closest_player = find_player_closest_to_ball(all_player_tracks, ball_tracks, ball_track["frame"])
        if closest_player:
            distance = np.sqrt(
                (closest_player["x"] - ball_track["x"])**2 +
                (closest_player["y"] - ball_track["y"])**2
            )
            if distance < 150:  # Threshold for interaction
                interaction_frames.append({
                    "frame": ball_track["frame"],
                    "player": closest_player,
                    "ball": ball_track,
                    "distance": distance
                })
                console_log(f"Interaction candidate at frame {ball_track['frame']}: distance = {distance:.1f}px")
    
    if not interaction_frames:
        console_log("No interaction frames found, using all player tracks", "warning")
        player_tracks = all_player_tracks
    else:
        # Filter to only tracks from the player closest to ball
        console_log(f"Found {len(interaction_frames)} interaction frames")
        interaction_frame_indices = {ifr["frame"] for ifr in interaction_frames}
        
        # Get player tracks that are close to interaction frames
        player_tracks = []
        for player_track in all_player_tracks:
            # Check if this player is near any interaction frame
            for ifr in interaction_frames:
                if abs(player_track["frame"] - ifr["frame"]) <= 5:
                    # Check if it's the same player (similar position)
                    distance = np.sqrt(
                        (player_track["x"] - ifr["player"]["x"])**2 +
                        (player_track["y"] - ifr["player"]["y"])**2
                    )
                    if distance < 100:  # Same player
                        player_tracks.append(player_track)
                        break
        
        console_log(f"Filtered to {len(player_tracks)} tracks from interacting player")
    
    # Step 3: Identify jump phase and calculate metrics
    console_log("\nSTEP 3: Identifying jump phase...")
    jump_start_frame = None
    jump_peak_frame = None
    jump_end_frame = None
    touch_frame = None
    
    if player_tracks and len(player_tracks) > 3:
        player_y_positions = [(t["frame"], t["y"]) for t in player_tracks]
        player_y_positions.sort(key=lambda x: x[0])  # Sort by frame
        
        # Find lowest Y (highest point in jump)
        min_y_frame, min_y = min(player_y_positions, key=lambda x: x[1])
        max_y_frame, max_y = max(player_y_positions, key=lambda x: x[1])
        
        console_log(f"Jump analysis: min Y (peak) at frame {min_y_frame} = {min_y:.1f}, max Y (ground) at frame {max_y_frame} = {max_y:.1f}")
        
        # Find jump start (when player starts moving up)
        for i in range(len(player_y_positions) - 1):
            if player_y_positions[i+1][1] < player_y_positions[i][1]:  # Moving up
                jump_start_frame = player_y_positions[i][0]
                console_log(f"Jump start detected at frame {jump_start_frame}")
                break
        
        jump_peak_frame = min_y_frame
        console_log(f"Jump peak at frame {jump_peak_frame}")
        
        # Find jump end (when player starts moving down after peak)
        peak_idx = next(i for i, (f, y) in enumerate(player_y_positions) if f == jump_peak_frame)
        for i in range(peak_idx, len(player_y_positions) - 1):
            if player_y_positions[i+1][1] > player_y_positions[i][1]:  # Moving down
                jump_end_frame = player_y_positions[i+1][0]
                console_log(f"Jump end detected at frame {jump_end_frame}")
                break
        
        # Find touch frame (when ball and player are closest)
        if ball_tracks:
            min_touch_distance = float('inf')
            for player_track in player_tracks:
                for ball_track in ball_tracks:
                    if abs(player_track["frame"] - ball_track["frame"]) <= 2:
                        distance = np.sqrt(
                            (player_track["x"] - ball_track["x"])**2 +
                            (player_track["y"] - ball_track["y"])**2
                        )
                        if distance < min_touch_distance:
                            min_touch_distance = distance
                            touch_frame = player_track["frame"]
            
            if touch_frame:
                console_log(f"Ball touch detected at frame {touch_frame}, distance: {min_touch_distance:.1f}px", "success")
    
    # Step 4: Create annotated frames with visual tracking
    console_log("\nSTEP 4: Creating annotated frames...")
    for frame_idx, frame in enumerate(frames):
        annotated = frame.copy()
        draw = ImageDraw.Draw(annotated)
        
        # Draw player tracks
        for player_track in player_tracks:
            if player_track["frame"] == frame_idx:
                box = player_track["box"]
                draw.rectangle(box, outline="green", width=3)
                draw.text((box[0], box[1] - 20), "PLAYER", fill="green")
                console_log(f"Frame {frame_idx}: Drawing player box {box}")
        
        # Draw ball tracks
        for ball_track in ball_tracks:
            if ball_track["frame"] == frame_idx:
                box = ball_track["box"]
                draw.ellipse(box, outline="red", width=3)
                draw.text((box[0], box[1] - 20), "BALL", fill="red")
                console_log(f"Frame {frame_idx}: Drawing ball box {box}")
        
        # Highlight key frames
        if frame_idx == jump_start_frame:
            draw.text((10, 10), "JUMP START", fill="yellow")
            console_log(f"Frame {frame_idx}: Marked as jump start")
        if frame_idx == jump_peak_frame:
            draw.text((10, 10), "JUMP PEAK", fill="yellow")
            console_log(f"Frame {frame_idx}: Marked as jump peak")
        if frame_idx == touch_frame:
            draw.text((10, 30), "BALL TOUCH", fill="orange")
            console_log(f"Frame {frame_idx}: Marked as ball touch")
        if frame_idx == jump_end_frame:
            draw.text((10, 50), "JUMP END", fill="yellow")
            console_log(f"Frame {frame_idx}: Marked as jump end")
        
        annotated_frames.append({
            "frame": frame_idx,
            "image": annotated,
            "has_player": any(t["frame"] == frame_idx for t in player_tracks),
            "has_ball": any(t["frame"] == frame_idx for t in ball_tracks)
        })
    
    console_log(f"Created {len(annotated_frames)} annotated frames")
    
    # Classify action type
    console_log("\nSTEP 5: Classifying action type...")
    action_type = classify_action(player_tracks, ball_tracks)
    console_log(f"Action type: {action_type}", "success")
    
    console_log("=" * 80)
    console_log("ANALYSIS COMPLETE")
    console_log(f"Player tracks: {len(player_tracks)}")
    console_log(f"Ball tracks: {len(ball_tracks)}")
    console_log(f"Jump: {jump_start_frame} -> {jump_peak_frame} -> {jump_end_frame}")
    console_log(f"Touch frame: {touch_frame}")
    console_log("=" * 80)
    
    return {
        "player_tracks": player_tracks,
        "ball_tracks": ball_tracks,
        "key_frames": key_frames,
        "player_detected": player_detected,
        "ball_detected": ball_detected,
        "action_type": action_type,
        "total_frames": len(frames),
        "jump_start_frame": jump_start_frame,
        "jump_peak_frame": jump_peak_frame,
        "jump_end_frame": jump_end_frame,
        "touch_frame": touch_frame,
        "annotated_frames": annotated_frames  # Will be converted to base64 in views
    }


def classify_action(player_tracks: List[Dict], ball_tracks: List[Dict]) -> str:
    """Classify the volleyball action."""
    if not player_tracks or not ball_tracks:
        return "other"
    
    player_y_positions = [track["y"] for track in player_tracks]
    ball_y_positions = [track["y"] for track in ball_tracks]
    
    if len(player_y_positions) < 3 or len(ball_y_positions) < 3:
        return "other"
    
    player_y_change = min(player_y_positions) - max(player_y_positions)
    ball_y_change = ball_y_positions[-1] - ball_y_positions[0]
    
    if player_y_change < -50 and ball_y_change > 0:
        return "kill"
    if abs(player_y_change) < 20 and ball_y_change < 0:
        return "block"
    
    return "other"


def calculate_metrics(analysis_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate volleyball performance metrics.
    Now includes hang time calculation.
    """
    console_log("=" * 80)
    console_log("CALCULATING METRICS")
    console_log("=" * 80)
    
    player_tracks = analysis_result.get("player_tracks", [])
    ball_tracks = analysis_result.get("ball_tracks", [])
    action_type = analysis_result.get("action_type", "other")
    jump_start = analysis_result.get("jump_start_frame")
    jump_peak = analysis_result.get("jump_peak_frame")
    jump_end = analysis_result.get("jump_end_frame")
    touch_frame = analysis_result.get("touch_frame")
    
    console_log(f"Player tracks: {len(player_tracks)}")
    console_log(f"Ball tracks: {len(ball_tracks)}")
    console_log(f"Jump frames: start={jump_start}, peak={jump_peak}, end={jump_end}")
    console_log(f"Touch frame: {touch_frame}")
    
    metrics = {
        "vertical_jump": None,
        "hang_time": None,
        "touch_point": None,
        "ball_touch_detected": False,
        "action_type": action_type
    }
    
    if not player_tracks:
        console_log("No player tracks, returning empty metrics", "warning")
        return metrics
    
    # Calculate vertical jump
    console_log("Calculating vertical jump...")
    player_y_positions = [track["y"] for track in player_tracks]
    if len(player_y_positions) > 1:
        min_y = min(player_y_positions)
        max_y = max(player_y_positions)
        jump_height_pixels = max_y - min_y
        
        console_log(f"Jump height: {jump_height_pixels:.1f} pixels")
        metrics["vertical_jump"] = round(jump_height_pixels * 0.1, 1)  # Convert to inches
        console_log(f"Vertical jump: {metrics['vertical_jump']} inches", "success")
    
    # Calculate hang time (time in air during jump)
    console_log("Calculating hang time...")
    if jump_start is not None and jump_end is not None:
        fps = 10.0  # Assuming 10 fps
        hang_time_frames = jump_end - jump_start
        hang_time_seconds = hang_time_frames / fps
        metrics["hang_time"] = round(hang_time_seconds, 2)
        console_log(f"Hang time: {metrics['hang_time']}s ({hang_time_frames} frames)", "success")
    elif jump_peak is not None:
        # Estimate from peak (half of total jump)
        fps = 10.0
        estimated_hang = 0.5  # Placeholder
        metrics["hang_time"] = estimated_hang
        console_log(f"Estimated hang time: {metrics['hang_time']}s", "warning")
    
    # Touch point
    console_log("Identifying touch point...")
    if touch_frame is not None:
        metrics["touch_point"] = {
            "frame": touch_frame,
            "timestamp": touch_frame / 10.0  # Assuming 10 fps
        }
        metrics["ball_touch_detected"] = True
        console_log(f"Touch point: frame {touch_frame} ({metrics['touch_point']['timestamp']:.2f}s)", "success")
    else:
        console_log("No touch point detected", "warning")
    
    console_log("=" * 80)
    console_log("METRICS CALCULATION COMPLETE")
    console_log(f"Metrics: {json.dumps(metrics, indent=2, default=str)}")
    console_log("=" * 80)
    
    return metrics
