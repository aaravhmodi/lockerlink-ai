"""
SAM3 Swing.mov Analysis Script
Player + Ball Tracking
Last Touch Detection
Jump Height Estimation
Ball Speed Estimation
Creates annotated output video.
"""

import os
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import ImageSequenceClip

import torch
from transformers import Sam3VideoModel, Sam3VideoProcessor
from transformers.video_utils import load_video


BASE_DIR = Path(__file__).resolve().parent

# -------------------------------
# Config
# -------------------------------
TOUCH_DISTANCE_PX = 80.0  # threshold for "ball touching player"

PLAYER_COLORS = [
    (0, 255, 0),    # green
    (0, 0, 255),    # blue
    (255, 255, 0),  # yellow
    (255, 0, 255),  # magenta
    (0, 255, 255),  # cyan
    (255, 128, 0),  # orange
    (128, 0, 255),  # purple
    (0, 128, 255)   # sky blue
]

BALL_COLOR = (255, 0, 0)  # bright red

PLAYER_ALPHA = 140
BALL_ALPHA = 200


# -------------------------------
# Utility functions
# -------------------------------

def center_of_box(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

def euclidean(a, b):
    return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

def draw_box(draw, box, color, label, font):
    x1, y1, x2, y2 = map(int, box)
    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
    w, h = draw.textsize(label, font=font)
    draw.rectangle([x1, y1 - h, x1 + w, y1], fill=color)
    draw.text((x1, y1 - h), label, fill="white", font=font)

def to_numpy_mask(mask):
    """
    SAM3 masks can be torch tensors or numpy; normalize to numpy bool array.
    Expect shape (H, W).
    """
    if mask is None:
        return None
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    # convert to bool
    if mask.dtype != bool:
        mask = mask > 0.5
    return mask

def overlay_mask(base_img, mask, color, alpha):
    """
    Overlay a single binary mask onto base_img (PIL.Image) with RGBA color.
    """
    mask = to_numpy_mask(mask)
    if mask is None:
        return base_img

    rgba = base_img.convert("RGBA")
    overlay = Image.new("RGBA", base_img.size, (*color, alpha))
    mask_img = Image.fromarray((mask.astype("uint8") * 255), mode="L")
    out = Image.composite(overlay, rgba, mask_img)
    return out


# -------------------------------
# MAIN
# -------------------------------

def main():

    # --------------------------
    # Load HF token
    # --------------------------
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("Missing HF_TOKEN")

    # --------------------------
    # Load model
    # --------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    model = Sam3VideoModel.from_pretrained(
        "facebook/sam3",
        token=token,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)

    processor = Sam3VideoProcessor.from_pretrained(
        "facebook/sam3",
        token=token
    )

    # --------------------------
    # Load video frames
    # --------------------------
    video_path = BASE_DIR / "swing.mov"
    frames, fps = load_video(str(video_path))
    print("Frames:", len(frames), "FPS:", fps)

    # --------------------------
    # Build SAM3 Video Session
    # --------------------------
    session = processor.init_video_session(
        video=frames,
        inference_device=device,
        processing_device="cpu",
        video_storage_device="cpu",
        dtype=torch.float16 if device == "cuda" else torch.float32,
    )

    # two text prompts: player + ball
    session = processor.add_text_prompt(session, "volleyball player")
    session = processor.add_text_prompt(session, "volleyball ball")

    # --------------------------
    # Run SAM3 across video
    # --------------------------
    outputs = {}
    print("Running SAM3 tracking...")
    for out in model.propagate_in_video_iterator(session):
        post = processor.postprocess_outputs(session, out)
        outputs[out.frame_idx] = post
        print("Processed frame", out.frame_idx + 1)

    # --------------------------
    # ANALYSIS / VISUALIZATION
    # --------------------------
    distances = []        # distance ball ↔ closest player each frame (if both present)
    ball_positions = []   # centers per frame or None
    player_positions = [] # matching player centers or None
    annotated_frames = []

    try:
        font = ImageFont.truetype("arial.ttf", 22)
    except:
        font = ImageFont.load_default()

    last_ball_center = None

    for i, frame in enumerate(frames):
        det = outputs.get(i, {})
        boxes = det.get("boxes", [])
        masks = det.get("masks", [])
        labels = det.get("text_prompts", [])

        # Normalize to Python lists
        if isinstance(boxes, torch.Tensor):
            boxes = boxes.detach().cpu().tolist()
        if isinstance(masks, torch.Tensor):
            masks = list(masks.detach().cpu())

        img = Image.fromarray(frame).convert("RGBA")

        # separate detections by type
        ball_indices = []
        player_indices = []

        for idx, label in enumerate(labels):
            label_l = str(label).lower()
            if "ball" in label_l:
                ball_indices.append(idx)
            else:
                player_indices.append(idx)

        # ---------- choose best ball detection (closest to last_ball_center) ----------
        chosen_ball_idx = None
        ball_center = None
        ball_box = None

        if ball_indices:
            if last_ball_center is None or len(ball_indices) == 1:
                chosen_ball_idx = ball_indices[0]
            else:
                # pick ball box whose center is closest to previous ball position
                min_prev_d = float("inf")
                best_idx = None
                for idx in ball_indices:
                    b = boxes[idx]
                    c = center_of_box(b)
                    d = euclidean(c, last_ball_center)
                    if d < min_prev_d:
                        min_prev_d = d
                        best_idx = idx
                chosen_ball_idx = best_idx

            ball_box = boxes[chosen_ball_idx]
            ball_center = center_of_box(ball_box)
            last_ball_center = ball_center

        # ---------- overlay masks ----------
        # players: different colors
        for j, p_idx in enumerate(player_indices):
            if p_idx < len(masks):
                color = PLAYER_COLORS[j % len(PLAYER_COLORS)]
                img = overlay_mask(img, masks[p_idx], color, PLAYER_ALPHA)

        # ball: single distinct color
        if chosen_ball_idx is not None and chosen_ball_idx < len(masks):
            img = overlay_mask(img, masks[chosen_ball_idx], BALL_COLOR, BALL_ALPHA)

        # convert back to RGB for drawing lines/text
        img_rgb = img.convert("RGB")
        draw = ImageDraw.Draw(img_rgb)

        # ---------- tracking: closest player to ball ----------
        tracked_player_center = None
        tracked_player_box = None
        frame_distance = None

        if ball_center is not None and player_indices:
            min_d = float("inf")
            best_box = None
            best_center = None

            for j, p_idx in enumerate(player_indices):
                pb = boxes[p_idx]
                pc = center_of_box(pb)
                d = euclidean(pc, ball_center)
                if d < min_d:
                    min_d = d
                    best_box = pb
                    best_center = pc

            frame_distance = min_d
            tracked_player_box = best_box
            tracked_player_center = best_center

            # only count as "touch" if within threshold
            if frame_distance <= TOUCH_DISTANCE_PX:
                distances.append(frame_distance)
                ball_positions.append(ball_center)
                player_positions.append(tracked_player_center)
            else:
                distances.append(None)
                ball_positions.append(None)
                player_positions.append(None)
        else:
            distances.append(None)
            ball_positions.append(None)
            player_positions.append(None)

        # ---------- draw boxes + line for visualization ----------
        if ball_box is not None:
            draw_box(draw, ball_box, BALL_COLOR, "BALL", font)

        if tracked_player_box is not None:
            color = (0, 255, 0) if (
                frame_distance is not None and frame_distance <= TOUCH_DISTANCE_PX
            ) else (0, 128, 255)
            label = "PLAYER (touch)" if (
                frame_distance is not None and frame_distance <= TOUCH_DISTANCE_PX
            ) else "PLAYER"
            draw_box(draw, tracked_player_box, color, label, font)

            if ball_center is not None:
                draw.line([ball_center, tracked_player_center],
                          fill="yellow", width=3)
                if frame_distance is not None:
                    txt = f"{frame_distance:.1f}px"
                    draw.text(ball_center, txt, fill="yellow", font=font)

        annotated_frames.append(img_rgb)

    # --------------------------
    # LAST TOUCH (minimum distance frame)
    # --------------------------
    # consider only frames where we had a valid touch (distance not None)
    valid_idxs = [idx for idx, d in enumerate(distances) if d is not None]
    if not valid_idxs:
        print("No valid touch frames found (ball never within threshold).")
        last_touch_frame = None
    else:
        best_idx = min(valid_idxs, key=lambda idx: distances[idx])
        last_touch_frame = best_idx
        print("Last touch frame (min distance within threshold):", last_touch_frame)

    # --------------------------
    # Convert px → cm using ball width at last-touch frame
    # --------------------------
    if last_touch_frame is not None:
        det_last = outputs[last_touch_frame]
        boxes_last = det_last.get("boxes", [])
        labels_last = det_last.get("text_prompts", [])
        # find ball box in that frame
        ball_idx_last = None
        for idx, lbl in enumerate(labels_last):
            if "ball" in str(lbl).lower():
                ball_idx_last = idx
                break

        if ball_idx_last is not None:
            ball_box_px = boxes_last[ball_idx_last]
            ball_width_px = ball_box_px[2] - ball_box_px[0]

            REAL_BALL_DIAMETER_CM = 21
            cm_per_px = REAL_BALL_DIAMETER_CM / ball_width_px
            print("cm per px =", cm_per_px)

            # --------------------------
            # JUMP HEIGHT (vertical)
            # --------------------------
            # use only frames with valid player position
            valid_player_ys = [p[1] for p in player_positions if p is not None]
            if valid_player_ys:
                jump_height_px = max(valid_player_ys) - min(valid_player_ys)
                jump_height_cm = jump_height_px * cm_per_px
                print("Estimated jump height:", round(jump_height_cm, 1), "cm")

            # --------------------------
            # Ball speed estimation
            # --------------------------
            valid_ball_centers = [b for b in ball_positions if b is not None]
            speeds = []
            for k in range(1, len(valid_ball_centers)):
                d = euclidean(valid_ball_centers[k], valid_ball_centers[k-1])
                speed_cm_s = (d * cm_per_px) * fps
                speeds.append(speed_cm_s)

            if speeds:
                avg_speed = np.mean(speeds)
                print("Estimated ball speed:", round(avg_speed, 1), "cm/s")
        else:
            print("Could not find ball box on last-touch frame; skipping cm conversion.")
    else:
        print("No last-touch frame; skipping jump height / speed.")

    # --------------------------
    # Save annotated video
    # --------------------------
    clip = ImageSequenceClip([np.array(f) for f in annotated_frames], fps=fps)
    out_path = BASE_DIR / "swing_sam3.mp4"
    clip.write_videofile(str(out_path), codec="libx264")

    print("Saved:", out_path)


if __name__ == "__main__":
    main()
