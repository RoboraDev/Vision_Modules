# rvm/detect/yolo.py
import torch
from ultralytics import YOLO
import numpy as np
from typing import List

from rvm.core.types import Box


class YOLODetector:
    """Wrapper for YOLOv11 detection."""

    def __init__(self, model_path: str = "yolo11n.pt"):
        print(f"Loading YOLO model from {model_path}...")
        self.model = YOLO(model_path)

    def detect(self, image: np.ndarray) -> List[Box]:
        """
        Run object detection on an image.

        Args:
            image (np.ndarray): Input BGR image.

        Returns:
            List[Box]: List of detection results.
        """
        results = self.model(image, verbose=False)
        boxes: List[Box] = []

        for r in results:
            for b in r.boxes:
                xyxy = b.xyxy[0].cpu().numpy().astype(int).tolist()
                conf = float(b.conf.cpu().numpy())
                cls = int(b.cls.cpu().numpy())
                boxes.append(Box(x1=xyxy[0], y1=xyxy[1], x2=xyxy[2], y2=xyxy[3],
                                 confidence=conf, class_id=cls))
        return boxes
