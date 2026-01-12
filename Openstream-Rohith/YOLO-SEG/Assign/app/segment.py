import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("Models/yolov8n-seg.pt")


def segment_image(inp, out):
    img = cv2.imread(inp)
    r = model(img, conf=0.4)[0]

    if r.masks is None:
        cv2.imwrite(out, img)
        return

    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.uint8)

    for m in r.masks.data:
        m = cv2.resize(m.cpu().numpy(), (w, h))
        mask |= (m > 0.5).astype(np.uint8)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=2)

    out_img = img * mask[:, :, None]
    cv2.imwrite(out, out_img)
