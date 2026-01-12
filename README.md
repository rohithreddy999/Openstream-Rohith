# Openstream Rohith

## Overview
This project implements a YOLO segmentation model for image and  video processing.

## Requirements
- Python 3.x
- Required packages listed in requirements.txt

## Setup
1. Clone the repository.
2. Navigate to the `Assign` directory.
3. Install the required packages using the command:
   `pip install -r requirements.txt`

## Running the Application
1. Run the main application using:
   `uvicorn app.main:app`
2. Upload files via endpoint fastapi :
   for images : http://localhost:8000/docs#/default/image_image_post
   for videos : http://localhost:8000/docs#/default/video_video_post
3. The output will be saved in the `output` directory.

## Model
The model used is `yolov8n-seg.pt`, located in the `Models` directory.
