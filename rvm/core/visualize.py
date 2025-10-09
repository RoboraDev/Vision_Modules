# rvm/core/visualize.py
"""
Visualization utilities for detection/segmentation results.

Functions:
- draw_boxes(image, boxes)
- draw_masks(image, masks)
- draw_markers(image, markers)
"""

import cv2 as cv
import numpy as np
from typing import List, Tuple

from rvm.core.types import Box, Mask, Marker, QRCode, BarCode
import random

_COLOR_MAP = {}


def _get_color_for_id(obj_id):
    """
    Return a consistent color for a given object ID.
    """
    if obj_id not in _COLOR_MAP:
        # Generate a random color
        random.seed(obj_id)
        _COLOR_MAP[obj_id] = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
        )
    return _COLOR_MAP[obj_id]


def draw_boxes(frame, detections, show_label: bool = True):
    annotated = frame.copy()

    for det in detections:
        if hasattr(det, "to_dict"):
            det = det.to_dict()

        # Support both detection and tracking dict formats
        if "x1" in det:
            x1, y1, x2, y2 = int(det["x1"]), int(det["y1"]), int(det["x2"]), int(det["y2"])
        elif "bbox" in det:
            x1, y1, x2, y2 = map(int, det["bbox"])
        else:
            continue

        # Color: use track_id if available, else class_id, else default green
        if "id" in det and det["id"] is not None and det["id"] >= 0:
            color = _get_color_for_id(det["id"])
        elif "class_id" in det:
            color = _get_color_for_id(det["class_id"])
        else:
            color = (0, 255, 0)

        # Draw rectangle
        cv.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Draw label text
        if show_label:
            label_parts = []
            if "id" in det and det["id"] is not None and det["id"] >= 0:
                label_parts.append(f"ID {det['id']}")
            if "label" in det:
                label_parts.append(str(det["label"]))
            elif "class_id" in det:
                label_parts.append(f"cls {det['class_id']}")
            if "conf" in det:
                label_parts.append(f"{det['conf']:.2f}")

            if label_parts:
                label = " | ".join(label_parts)
                (tw, th), baseline = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv.rectangle(annotated, (x1, y1 - th - baseline), (x1 + tw, y1), color, -1)
                cv.putText(annotated, label, (x1, y1 - baseline),
                            cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)

    return annotated


def draw_masks(image: np.ndarray, masks: List[Mask], alpha: float = 0.4) -> np.ndarray:
    """ 
    Olay masks on the image with transparency.
    Each mask gets a unique color based on its segmentation hash.
    """
    img = image.copy()
    overlay = img.copy()

    for mask in masks:
        if len(mask.segmentation) == 0:
            continue
        obj_id = hash(tuple(map(tuple, mask.segmentation))) % 10000
        color = _get_color_for_id(obj_id)

        pts = np.array(mask.segmentation, dtype=np.int32).reshape((-1, 1, 2))
        cv.fillPoly(overlay, [pts], color)
        cv.polylines(img, [pts], True, color, 2)
        cv.putText(img, f"{mask.confidence:.2f}",
                    (pts[0][0][0], pts[0][0][1] - 5),
                    cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv.LINE_AA)

    return cv.addWeighted(overlay, alpha, img, 1 - alpha, 0)


def draw_markers(image: np.ndarray, markers: List[Marker]) -> np.ndarray:
    """
    Draw detected markers on the image.
    Each marker gets a unique color based on its ID or corners hash.
    """
    img = image.copy()
    for marker in markers:
        # Use marker ID if available, else hash corners for color
        obj_id = marker.id if marker.id is not None else hash(tuple(map(tuple, marker.corners))) % 10000
        color = _get_color_for_id(obj_id)

        corners = np.array(marker.corners, dtype=np.int32).reshape((-1, 1, 2))
        cv.polylines(img, [corners], True, color, 2)
        cX = int(np.mean([pt[0] for pt in marker.corners]))
        cY = int(np.mean([pt[1] for pt in marker.corners]))
        cv.putText(img, str(marker.id), (cX, cY),
                    cv.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv.LINE_AA)
    return img


def draw_qr_codes(image: np.ndarray, qr_codes: List[QRCode]) -> np.ndarray:
    """
    Draw detected QR codes on the image with text at top-right of bounding box.
    """
    img = image.copy()
    for i, qr_code in enumerate(qr_codes):
        # Use index-based color for QR codes
        color = _get_color_for_id(i + 1000)  # Offset to differentiate from markers
        
        # Draw bounding polygon
        corners = np.array(qr_code.corners, dtype=np.int32).reshape((-1, 1, 2))
        cv.polylines(img, [corners], True, color, 2)
        
        # Calculate top-right position for text
        max_x = max(pt[0] for pt in qr_code.corners)
        min_y = min(pt[1] for pt in qr_code.corners)
        
        # Add padding to avoid overlap with bounding box
        text_x = max_x + 5
        text_y = min_y - 5
        
        # Draw QR label at top-right
        cv.putText(img, "QR", (text_x, text_y),
                    cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv.LINE_AA)
        
        # Draw truncated data below the label
        data_text = qr_code.data[:20] + "..." if len(qr_code.data) > 20 else qr_code.data
        cv.putText(img, data_text, (text_x, text_y + 20),
                    cv.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv.LINE_AA)
    
    return img


def draw_barcodes(image: np.ndarray, barcodes: List[BarCode]) -> np.ndarray:
    """
    Draw detected barcodes on the image with text at top-right of bounding box.
    """
    img = image.copy()
    for i, barcode in enumerate(barcodes):
        # Use index-based color for barcodes
        color = _get_color_for_id(i + 2000)  # Different offset for barcodes
        
        # Draw bounding polygon
        corners = np.array(barcode.corners, dtype=np.int32).reshape((-1, 1, 2))
        cv.polylines(img, [corners], True, color, 2)
        
        # Calculate top-right position for text
        max_x = max(pt[0] for pt in barcode.corners)
        min_y = min(pt[1] for pt in barcode.corners)
        
        # Add padding to avoid overlap with bounding box
        text_x = max_x + 5
        text_y = min_y - 5
        
        # Draw barcode label at top-right
        cv.putText(img, "BC", (text_x, text_y),
                    cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv.LINE_AA)
        
        # Draw truncated data below the label
        data_text = barcode.data[:15] + "..." if len(barcode.data) > 15 else barcode.data
        cv.putText(img, data_text, (text_x, text_y + 20),
                    cv.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv.LINE_AA)
    
    return img


def draw_pose_axes(image: np.ndarray, rvec, tvec, camera_matrix, dist_coeffs, length: float = 0.05):
    """
    Draw 3D coordinate axes for the given pose.
    """
    img = image.copy()
    try:
        cv.drawFrameAxes(   # <--- sửa cv2 -> cv
            img,
            np.array(camera_matrix, dtype=float),
            np.array(dist_coeffs, dtype=float),
            np.array(rvec, dtype=float).reshape(3, 1),
            np.array(tvec, dtype=float).reshape(3, 1),
            length
        )
    except Exception as e:
        print(f"[WARN] draw_pose_axes failed: {e}")  # để dễ debug
    return img
