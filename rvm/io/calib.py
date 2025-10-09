"""
Camera calibration utilities.
Supports loading intrinsics from YAML or JSON files.
"""
import json
import yaml
import numpy as np
from pathlib import Path


def load_camera_calibration(path: str):
    """Load camera matrix (K) and distortion coefficients (dist)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Camera calibration file not found: {path}")

    if p.suffix.lower() in [".yml", ".yaml"]:
        data = yaml.safe_load(p.read_text())
    else:
        data = json.loads(p.read_text())

    K = np.array(data.get("camera_matrix") or data.get("K"))
    dist = np.array(data.get("dist_coeffs") or data.get("dist") or [])
    return K, dist


def save_camera_calibration(path: str, camera_matrix: np.ndarray, dist_coeffs: np.ndarray):
    """Save calibration to YAML."""
    data = {"camera_matrix": camera_matrix.tolist(), "dist_coeffs": dist_coeffs.tolist()}
    yaml.safe_dump(data, open(path, "w"))
