import numpy as np
from rvm.markers.pose import PoseEstimator
from rvm.core.types import Marker


def test_estimator_basic():
    corners = np.array([[100, 100], [200, 100], [200, 200], [100, 200]], dtype=np.float32)
    marker = Marker(id=1, corners=corners.tolist())
    K = np.array([[300, 0, 160], [0, 300, 120], [0, 0, 1]], dtype=float)
    dist = np.zeros((5,))

    estimator = PoseEstimator(marker_size_m=0.1)
    poses = estimator.estimate_from_markers([marker], K, dist)

    assert len(poses) == 1
    assert hasattr(poses[0], "rvec")
    assert isinstance(poses[0].success, bool)
