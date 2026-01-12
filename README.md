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
1. Place your input video files in the `input` directory.
2. Run the main application using:
   `uvicorn app.main:app`
3. The output will be saved in the `output` directory.

## Model
The model used is `yolov8n-seg.pt`, located in the `Models` directory.
