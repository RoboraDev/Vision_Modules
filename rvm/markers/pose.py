"""
Pose estimation for ArUco or custom markers.
Computes 6DoF (rvec, tvec) from detected marker corners using solvePnP.
"""
import cv2
import numpy as np
from typing import List
from rvm.core.types import Pose, Marker


class PoseEstimator:
    def __init__(self, marker_size_m: float = 0.05, solve_method=cv2.SOLVEPNP_IPPE_SQUARE):
        """
        Args:
            marker_size_m: real-world side length of marker (in meters)
            solve_method: OpenCV solvePnP flag (default: IPPE for planar square)
        """
        self.marker_size = float(marker_size_m)
        self.solve_method = solve_method

        s = self.marker_size / 2.0
        self.obj_points = np.array([
            [-s,  s, 0.0],
            [ s,  s, 0.0],
            [ s, -s, 0.0],
            [-s, -s, 0.0],
        ], dtype=np.float32)

    def estimate_from_markers(self, markers: List[Marker], camera_matrix: np.ndarray, dist_coeffs: np.ndarray):
        poses = []
        for marker in markers:
            img_pts = np.array(marker.corners, dtype=np.float32)
            try:
                success, rvec, tvec = cv2.solvePnP(
                    self.obj_points, img_pts, camera_matrix, dist_coeffs, flags=self.solve_method
                )
            except Exception:
                success, rvec, tvec = False, None, None

            if not success:
                poses.append(Pose(marker_id=marker.id, rvec=[0,0,0], tvec=[0,0,0], T=None, success=False))
                continue

            R, _ = cv2.Rodrigues(rvec)
            T = np.eye(4, dtype=float)
            T[:3, :3] = R
            T[:3, 3] = tvec.reshape(3)

            poses.append(Pose(
                marker_id=marker.id,
                rvec=rvec.reshape(3).tolist(),
                tvec=tvec.reshape(3).tolist(),
                T=T.tolist(),
                success=True
            ))
        return poses
