# rvm/track/tracker.py
from typing import List, Dict
from ultralytics import YOLO
import numpy as np


class IoUTracker:
    """
    Simple IoU-based tracker (lightweight).
    Assigns IDs across frames using IoU matching.
    """
    def __init__(self, iou_thresh: float = 0.5):
        self.iou_thresh = iou_thresh
        self.next_id = 0
        self.tracks = []  # [{ "bbox": [...], "id": int }]

    def _iou(self, boxA, boxB):
        # box = [x1, y1, x2, y2]
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

        return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

    def update(self, detections):
        """
        Args:
            detections: list of dict 
                - {"bbox":[x1,y1,x2,y2], "score":float, "class":int}

        Returns:
            list of dict: [{"x1":..,"y1":..,"x2":..,"y2":..,
                            "confidence":..,"class_id":..,"track_id":..}]
        """
        updated_tracks = []

        for det in detections:
            if "bbox" not in det:
                continue
            box = det["bbox"]
            conf = det.get("score", 1.0)
            cls_id = det.get("class", -1)

            # Match with existing tracks by IoU
            best_iou, best_track = 0, None
            for tr in self.tracks:
                iou_val = self._iou(box, tr["bbox"])
                if iou_val > best_iou:
                    best_iou, best_track = iou_val, tr

            if best_iou > self.iou_thresh:
                track_id = best_track["id"]
                best_track["bbox"] = box  # update position
            else:
                track_id = self.next_id
                self.next_id += 1
                self.tracks.append({"id": track_id, "bbox": box})

            updated_tracks.append({
            "bbox": [int(box[0]), int(box[1]), int(box[2]), int(box[3])],
            "id": track_id,
            "conf": float(conf),
            "label": str(cls_id)
        })

        return updated_tracks


class UltralyticsTracker:
    """
    Wrapper cho Ultralytics YOLO tracking.
    Return list dict: {"bbox": [...], "id": int, "conf": float, "label": str}.
    """
    def __init__(self, model_path: str = "yolo11n.pt"):
        self.model = YOLO(model_path)

    def update(self, frame: np.ndarray) -> List[Dict]:
        results = self.model.track(frame, persist=True, verbose=False)
        tracks = []
        if len(results) > 0:
            r = results[0]  # only one frame
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                track_id = int(box.id.item()) if box.id is not None else -1
                label = self.model.names[cls_id]
                tracks.append({
                    "bbox": [x1, y1, x2, y2],
                    "id": track_id,
                    "conf": conf,
                    "label": label
                })
        return tracks
