"""
SAM3 Swing.mov Analysis Script
Tracks the player that touches the volleyball and creates an annotated video.
Usage: python test_swing.py
"""

import os
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2

# Set BASE_DIR first
BASE_DIR = Path(__file__).resolve().parent

# Load environment variables from .env.local
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=BASE_DIR / '.env.local')
    load_dotenv(dotenv_path=BASE_DIR / '.env')
except ImportError:
    pass  # Will load manually if dotenv not available

# Add app directory to path
sys.path.insert(0, str(BASE_DIR))

from app.sam3_inference import segment_image, get_hf_token
from app.video_utils import extract_frames, create_annotated_video


def calculate_center(box):
    """Calculate center point of a bounding box [x1, y1, x2, y2]."""
    if box is None or len(box) < 4:
        return None
    x1, y1, x2, y2 = box[:4]
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def calculate_distance(point1, point2):
    """Calculate Euclidean distance between two points."""
    if point1 is None or point2 is None:
        return float('inf')
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def find_closest_player_to_ball(player_results, ball_results):
    """
    Find the player closest to the ball.
    
    Args:
        player_results: Dict with 'boxes', 'masks', 'scores' from SAM3
        ball_results: Dict with 'boxes', 'masks', 'scores' from SAM3
    
    Returns:
        Dict with closest player info: {'box': [...], 'mask': ..., 'score': ..., 'distance': ...}
    """
    # Check if boxes exist (handle None and tensors properly)
    player_boxes_raw = player_results.get("boxes")
    ball_boxes_raw = ball_results.get("boxes")
    
    if player_boxes_raw is None or ball_boxes_raw is None:
        return None
    
    # Convert tensors to lists if needed
    if hasattr(player_boxes_raw, 'tolist'):
        player_boxes = player_boxes_raw.tolist()
    elif hasattr(player_boxes_raw, 'cpu'):
        player_boxes = player_boxes_raw.cpu().tolist()
    else:
        player_boxes = player_boxes_raw
    
    if hasattr(ball_boxes_raw, 'tolist'):
        ball_boxes = ball_boxes_raw.tolist()
    elif hasattr(ball_boxes_raw, 'cpu'):
        ball_boxes = ball_boxes_raw.cpu().tolist()
    else:
        ball_boxes = ball_boxes_raw
    
    # Check if we have valid boxes
    if not player_boxes or not ball_boxes:
        return None
    
    # Handle empty lists
    if len(player_boxes) == 0 or len(ball_boxes) == 0:
        return None
    
    # Get ball center (use first ball detection)
    first_ball_box = ball_boxes[0]
    if isinstance(first_ball_box, list):
        ball_box_list = first_ball_box
    elif hasattr(first_ball_box, 'tolist'):
        ball_box_list = first_ball_box.tolist()
    elif hasattr(first_ball_box, 'cpu'):
        ball_box_list = first_ball_box.cpu().tolist()
    else:
        ball_box_list = first_ball_box
    
    ball_center = calculate_center(ball_box_list)
    if ball_center is None:
        return None
    
    # Find closest player
    closest_player = None
    min_distance = float('inf')
    
    for i, player_box in enumerate(player_boxes):
        # Convert box to list format
        if isinstance(player_box, list):
            box = player_box
        elif hasattr(player_box, 'tolist'):
            box = player_box.tolist()
        elif hasattr(player_box, 'cpu'):
            box = player_box.cpu().tolist()
        else:
            box = player_box
        
        player_center = calculate_center(box)
        if player_center is None:
            continue
        
        distance = calculate_distance(player_center, ball_center)
        if distance < min_distance:
            min_distance = distance
            closest_player = {
                'box': box,
                'index': i,
                'distance': distance,
                'center': player_center
            }
            
            # Add mask and score if available
            masks_raw = player_results.get("masks")
            if masks_raw is not None:
                try:
                    if hasattr(masks_raw, '__getitem__'):
                        if i < len(masks_raw):
                            closest_player['mask'] = masks_raw[i]
                except:
                    pass
            
            scores_raw = player_results.get("scores")
            if scores_raw is not None:
                try:
                    if hasattr(scores_raw, 'tolist'):
                        scores = scores_raw.tolist()
                    elif hasattr(scores_raw, 'cpu'):
                        scores = scores_raw.cpu().tolist()
                    else:
                        scores = scores_raw
                    
                    if isinstance(scores, list) and i < len(scores):
                        closest_player['score'] = scores[i]
                except:
                    pass
    
    return closest_player


def draw_annotations(frame, player_result, ball_result, tracked_player=None, frame_num=0):
    """
    Draw annotations on a frame: bounding boxes, masks, labels.
    
    Args:
        frame: PIL Image
        player_result: SAM3 result for players
        ball_result: SAM3 result for ball
        tracked_player: Dict with tracked player info (closest to ball)
        frame_num: Frame number for display
    
    Returns:
        Annotated PIL Image
    """
    # Create a copy to draw on
    annotated = frame.copy()
    draw = ImageDraw.Draw(annotated)
    
    try:
        # Try to load a font (fallback to default if not available)
        try:
            font = ImageFont.truetype("arial.ttf", 16)
            font_large = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()
            font_large = ImageFont.load_default()
        
        # Draw frame number
        draw.text((10, 10), f"Frame {frame_num + 1}", fill="white", font=font_large)
        
        # Draw ball detections (red)
        if ball_result.get("boxes") is not None:
            ball_boxes = ball_result["boxes"]
            if hasattr(ball_boxes, 'tolist'):
                ball_boxes = ball_boxes.tolist()
            elif hasattr(ball_boxes, 'cpu'):
                ball_boxes = ball_boxes.cpu().tolist()
            
            for i, box in enumerate(ball_boxes[:3]):  # Limit to top 3
                if isinstance(box, list):
                    bbox = box
                else:
                    bbox = box.tolist() if hasattr(box, 'tolist') else box
                
                if len(bbox) >= 4:
                    x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
                    # Draw bounding box
                    draw.rectangle([x1, y1, x2, y2], outline="red", width=4)
                    # Draw label with background
                    label = "Ball"
                    draw.rectangle([x1, y1 - 25, x1 + 60, y1], fill="red", outline="red")
                    draw.text((x1 + 5, y1 - 22), label, fill="white", font=font)
        
        # Draw all player detections
        player_boxes_raw = player_result.get("boxes")
        if player_boxes_raw is not None:
            # Convert to list safely
            if hasattr(player_boxes_raw, 'tolist'):
                player_boxes = player_boxes_raw.tolist()
            elif hasattr(player_boxes_raw, 'cpu'):
                player_boxes = player_boxes_raw.cpu().tolist()
            else:
                player_boxes = player_boxes_raw
            
            if isinstance(player_boxes, list) and len(player_boxes) > 0:
                for i, box in enumerate(player_boxes[:10]):  # Limit to top 10
                    # Convert box to list
                    if isinstance(box, list):
                        bbox = box
                    elif hasattr(box, 'tolist'):
                        bbox = box.tolist()
                    elif hasattr(box, 'cpu'):
                        bbox = box.cpu().tolist()
                    else:
                        bbox = box
                
                if len(bbox) >= 4:
                    x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
                    
                    # Highlight tracked player (closest to ball) in green
                    if tracked_player and tracked_player.get('index') == i:
                        # Draw thick green box for tracked player
                        draw.rectangle([x1, y1, x2, y2], outline="green", width=6)
                        label = f"TRACKED (d={tracked_player['distance']:.0f}px)"
                        # Draw label with background
                        text_bbox = draw.textbbox((x1, y1 - 30), label, font=font)
                        draw.rectangle(text_bbox, fill="green", outline="green")
                        draw.text((x1 + 3, y1 - 27), label, fill="white", font=font)
                    else:
                        # Draw blue box for other players
                        draw.rectangle([x1, y1, x2, y2], outline="blue", width=2)
                        label = f"Player {i+1}"
                        draw.text((x1, y1 - 20), label, fill="blue", font=font)
        
        # Draw distance line if tracked player exists
        if tracked_player:
            ball_boxes_raw = ball_result.get("boxes")
            if ball_boxes_raw is not None:
                # Convert to list safely
                if hasattr(ball_boxes_raw, 'tolist'):
                    ball_boxes = ball_boxes_raw.tolist()
                elif hasattr(ball_boxes_raw, 'cpu'):
                    ball_boxes = ball_boxes_raw.cpu().tolist()
                else:
                    ball_boxes = ball_boxes_raw
                
                if isinstance(ball_boxes, list) and len(ball_boxes) > 0:
                    ball_box = ball_boxes[0]
                    # Convert box to list
                    if isinstance(ball_box, list):
                        ball_bbox = ball_box
                    elif hasattr(ball_box, 'tolist'):
                        ball_bbox = ball_box.tolist()
                    elif hasattr(ball_box, 'cpu'):
                        ball_bbox = ball_box.cpu().tolist()
                    else:
                        ball_bbox = ball_box
                
                if len(ball_bbox) >= 4:
                    ball_center = calculate_center(ball_bbox)
                    player_center = tracked_player.get('center')
                    
                    if ball_center and player_center:
                        # Draw line between player and ball
                        draw.line([player_center, ball_center], fill="yellow", width=3)
                        # Draw distance text with background
                        mid_x = int((player_center[0] + ball_center[0]) / 2)
                        mid_y = int((player_center[1] + ball_center[1]) / 2)
                        dist_text = f"{tracked_player['distance']:.0f}px"
                        text_bbox = draw.textbbox((mid_x, mid_y), dist_text, font=font)
                        draw.rectangle(text_bbox, fill="yellow", outline="yellow")
                        draw.text((mid_x, mid_y), dist_text, fill="black", font=font)
    
    except Exception as e:
        print(f"  ⚠ Warning: Error drawing annotations: {e}")
        import traceback
        traceback.print_exc()
    
    return annotated


def main():
    print("=" * 80)
    print("SAM3 Swing.mov Player Tracking & Video Generation")
    print("=" * 80)
    
    # Load token
    token = None
    token = os.environ.get('HF_TOKEN')
    
    if not token:
        env_local = BASE_DIR / '.env.local'
        if env_local.exists():
            print(f"Loading token from: {env_local}")
            try:
                with open(env_local, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith('HF_TOKEN='):
                            token = line.split('=', 1)[1].strip()
                            token = token.strip('"').strip("'")
                            os.environ['HF_TOKEN'] = token
                            print(f"✓ Loaded token from .env.local")
                            break
            except Exception as e:
                print(f"Warning: Could not read .env.local: {e}")
    
    if not token:
        token = get_hf_token()
    
    if not token:
        print("\nERROR: HF_TOKEN not found!")
        print(f"Please set HF_TOKEN in .env.local or as environment variable")
        return 1
    
    print(f"✓ HF_TOKEN found: {token[:10]}...{token[-4:]}")
    
    # Find swing.mov
    possible_paths = ['swing.mov', 'app/swing.mov', 'media/swing.mov']
    swing_path = None
    
    for path in possible_paths:
        if os.path.exists(path):
            swing_path = path
            break
    
    if not swing_path:
        print(f"\nERROR: swing.mov not found!")
        print(f"Searched in: {possible_paths}")
        print("Please place swing.mov in the project root or app/ directory")
        return 1
    
    print(f"✓ Found swing.mov at: {swing_path}")
    
    # Extract frames
    print("\n" + "=" * 80)
    print("Step 1: Extracting frames from swing.mov...")
    print("=" * 80)
    
    try:
        frames = extract_frames(swing_path, fps=10, max_frames=None)  # Process all frames
        print(f"✓ Extracted {len(frames)} frames")
    except Exception as e:
        print(f"ERROR extracting frames: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Analyze frames and track player
    print("\n" + "=" * 80)
    print("Step 2: Analyzing frames and tracking player closest to ball...")
    print("=" * 80)
    
    results = []
    annotated_frames = []
    tracked_player_history = []
    
    import time
    start_time = time.time()
    
    for i, frame in enumerate(frames):
        frame_start = time.time()
        print(f"\nProcessing frame {i+1}/{len(frames)}...")
        
        try:
            # Detect all players
            print(f"  → Detecting players...")
            player_result = segment_image(frame, text_prompt="a volleyball player")
            
            # Detect ball
            print(f"  → Detecting ball...")
            ball_result = segment_image(frame, text_prompt="a volleyball ball")
            
            # Find player closest to ball
            tracked_player = find_closest_player_to_ball(player_result, ball_result)
            
            if tracked_player:
                print(f"  ✓ Tracked player found: distance={tracked_player['distance']:.1f}px, score={tracked_player.get('score', 'N/A')}")
                tracked_player_history.append({
                    'frame': i,
                    'player': tracked_player,
                    'ball_detected': ball_result.get("masks") is not None
                })
            else:
                print(f"  ⚠ No tracked player found (no ball or players detected)")
            
            # Create annotated frame
            annotated_frame = draw_annotations(frame, player_result, ball_result, tracked_player, frame_num=i)
            annotated_frames.append({
                'frame': i,
                'image': annotated_frame
            })
            
            # Store results
            player_detected = player_result.get("masks") is not None
            ball_detected = ball_result.get("masks") is not None
            
            frame_time = time.time() - frame_start
            print(f"  ⏱ Frame {i+1} processed in {frame_time:.2f}s")
            
            results.append({
                "frame": i,
                "player_detected": player_detected,
                "ball_detected": ball_detected,
                "tracked_player": tracked_player is not None,
                "player_distance": tracked_player['distance'] if tracked_player else None,
                "processing_time": frame_time,
            })
            
        except Exception as e:
            frame_time = time.time() - frame_start
            print(f"  ✗ ERROR processing frame {i+1} (after {frame_time:.2f}s): {e}")
            import traceback
            traceback.print_exc()
            
            # Use original frame if annotation fails
            annotated_frames.append({
                'frame': i,
                'image': frame
            })
            
            results.append({
                "frame": i,
                "error": str(e),
                "processing_time": frame_time,
            })
            
            # Clear memory on error
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except:
                pass
    
    # Create annotated video
    print("\n" + "=" * 80)
    print("Step 3: Creating annotated video...")
    print("=" * 80)
    
    try:
        output_video_path = os.path.join(BASE_DIR, "swing_tracked.mp4")
        print(f"Creating annotated video: {output_video_path}")
        print(f"  → {len(annotated_frames)} frames to process...")
        
        # Get original video FPS
        cap = cv2.VideoCapture(swing_path)
        original_fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
        print(f"  → Using FPS: {original_fps:.2f}")
        cap.release()
        
        create_annotated_video(annotated_frames, output_video_path, fps=original_fps)
        print(f"✓ Annotated video created successfully!")
        print(f"  → Output: {output_video_path}")
        print(f"  → File size: {os.path.getsize(output_video_path) / (1024*1024):.2f} MB")
    except Exception as e:
        print(f"✗ ERROR creating video: {e}")
        import traceback
        traceback.print_exc()
        output_video_path = None
    
    # Summary
    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    total_frames = len(results)
    player_detections = sum(1 for r in results if r.get("player_detected", False))
    ball_detections = sum(1 for r in results if r.get("ball_detected", False))
    tracked_detections = sum(1 for r in results if r.get("tracked_player", False))
    successful_frames = sum(1 for r in results if "error" not in r)
    avg_time = sum(r.get("processing_time", 0) for r in results if "processing_time" in r) / max(successful_frames, 1)
    
    print(f"Total frames processed: {total_frames}")
    print(f"Successful frames: {successful_frames}/{total_frames}")
    print(f"Total processing time: {total_time:.2f}s")
    print(f"Average time per frame: {avg_time:.2f}s")
    if total_frames > 0:
        print(f"Player detected in: {player_detections}/{total_frames} frames ({player_detections/total_frames*100:.1f}%)")
        print(f"Ball detected in: {ball_detections}/{total_frames} frames ({ball_detections/total_frames*100:.1f}%)")
        print(f"Tracked player (closest to ball) found in: {tracked_detections}/{total_frames} frames ({tracked_detections/total_frames*100:.1f}%)")
    
    if tracked_player_history:
        avg_distance = np.mean([t['player']['distance'] for t in tracked_player_history])
        min_distance = min([t['player']['distance'] for t in tracked_player_history])
        print(f"\nTracking Statistics:")
        print(f"  Average distance to ball: {avg_distance:.1f}px")
        print(f"  Minimum distance to ball: {min_distance:.1f}px")
        print(f"  Frames with ball contact: {sum(1 for t in tracked_player_history if t['ball_detected'] and t['player']['distance'] < 100)}")
    
    if output_video_path and os.path.exists(output_video_path):
        print(f"\n✓ Output video saved: {output_video_path}")
        print(f"  → You can now view the tracked player visualization!")
    else:
        print(f"\n⚠ Output video was not created")
    
    print("=" * 80)
    print("Analysis complete!")
    print("=" * 80)
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
