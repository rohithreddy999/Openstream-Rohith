import cv2
import numpy as np
import subprocess
import os
import shutil
import logging
from ultralytics import YOLO

logger = logging.getLogger(__name__)

# Load YOLOv8 segmentation model
model = YOLO("Models/yolov8n-seg.pt")


def _faststart_mp4(path: str):

    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        logger.warning("ffmpeg not found; skipping faststart for %s", path)
        return

    temp = path.replace(".mp4", "_fs.mp4")
    try:
        subprocess.run(
            [
                ffmpeg_bin,
                "-y",
                "-i",
                path,
                "-movflags",
                "faststart",
                "-codec",
                "copy",
                temp,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        os.replace(temp, path)
    except subprocess.CalledProcessError:
        logger.exception("ffmpeg faststart failed for %s", path)
        if os.path.exists(temp):
            os.remove(temp)


def segment_video(inp: str, out: str):
    cap = cv2.VideoCapture(inp)
    if not cap.isOpened():
        raise RuntimeError("Cannot open input video")

    # Input video properties
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    logger.info(f"Video: {w}x{h} @ {fps:.2f}fps, {total_frames} frames")

    # Output writer 
    writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    if not writer.isOpened():
        raise RuntimeError("VideoWriter failed to open")

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 30 == 0:  
            logger.info(f"Processing frame {frame_count}/{total_frames}")

        res = model(frame, conf=0.4, verbose=False)[0]

        # BLACK background
        black_bg = np.zeros_like(frame)

        if res.masks is None or len(res.masks.data) == 0:
            writer.write(black_bg)
            continue

        # Combine ALL detected person masks (not just first one)
        combined_mask = np.zeros((h, w), dtype=np.uint8)
        for mask_data in res.masks.data:
            mask = cv2.resize(mask_data.cpu().numpy(), (w, h))
            mask = (mask > 0.5).astype(np.uint8)
            combined_mask = np.maximum(combined_mask, mask)

        kernel = np.ones((3, 3), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)

        # Composite: person(s) + black background
        out_frame = np.where(combined_mask[:, :, None] == 1, frame, black_bg)
        writer.write(out_frame)

    cap.release()
    writer.release()

    logger.info(f"Processed {frame_count} frames")


    _faststart_mp4(out)
