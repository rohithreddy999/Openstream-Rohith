from fastapi import FastAPI, UploadFile
import shutil
import logging

from app.segment import segment_image
from app.video_segment import segment_video
import cv2, numpy as np
from ultralytics import YOLO
from app.utils import gradient

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

app = FastAPI(
    title="YOLO Segmentation API",
    description="AI-powered background removal for images and videos",
    version="1.0.0",
)


@app.get("/")
async def root():
    return {
        "message": "YOLO Segmentation Service",
        "version": "1.0.0",
        "endpoints": {
            "POST /image": "Upload image for background removal",
            "POST /video": "Upload video for background removal",
            "GET /docs": "Interactive API documentation",
            "GET /redoc": "Alternative API documentation",
        },
        "status": "Running",
        "model": "YOLOv8n-seg",
    }


@app.post("/image")
async def image(file: UploadFile):
    inp = f"input/{file.filename}"
    with open(inp, "wb") as f:
        shutil.copyfileobj(file.file, f)

    out = "output/result.png"
    segment_image(inp, out)
    return {"output": out}


@app.post("/video")
async def video(file: UploadFile):
    inp = f"input/{file.filename}"
    with open(inp, "wb") as f:
        shutil.copyfileobj(file.file, f)

    out = "output/result.mp4"
    segment_video(inp, out)
    return {"output": out}
