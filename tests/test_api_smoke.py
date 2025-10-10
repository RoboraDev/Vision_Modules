import numpy as np
import pytest
import tempfile
import cv2

from rvm import api


@pytest.mark.smoke
def test_detect_image_smoke(tmp_path):
    # Create a dummy image
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    img_path = tmp_path / "dummy.jpg"
    cv2.imwrite(str(img_path), img)

    results = api.detect(str(img_path), model="yolov8n.pt", out_dir=tmp_path)
    assert isinstance(results, list)


@pytest.mark.smoke
def test_segment_image_smoke(tmp_path):
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    img_path = tmp_path / "dummy.jpg"
    cv2.imwrite(str(img_path), img)

    results = api.segment_image(str(img_path), out_dir=tmp_path)
    assert isinstance(results, list)


@pytest.mark.smoke
def test_detect_markers_smoke(tmp_path):
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    img_path = tmp_path / "dummy.jpg"
    cv2.imwrite(str(img_path), img)

    results = api.detect_markers(str(img_path), out_dir=tmp_path)
    assert isinstance(results, list)
