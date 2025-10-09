# rvm/core/types.py
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional


@dataclass
class Box:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    class_id: int
    track_id: int = -1  # Optional, for tracking

    def to_dict(self):
        """Convert Box object to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class Mask:
    segmentation: List[List[int]]
    confidence: float
    class_id: int = 0

    def to_dict(self):
        return asdict(self)


@dataclass
class Marker:
    id: int
    corners: List[Tuple[int, int]]

    def to_dict(self):
        return asdict(self)


@dataclass
class QRCode:
    data: str
    corners: List[Tuple[int, int]]

    def to_dict(self):
        return asdict(self)
    
@dataclass
class BarCode:
    data: str
    corners: List[Tuple[int, int]]

    def to_dict(self):
        return asdict(self)

@dataclass
class Pose:
    marker_id: int
    rvec: List[float]
    tvec: List[float]
    T: Optional[List[List[float]]] = None
    success: bool = False

    def to_dict(self):
        return asdict(self)
