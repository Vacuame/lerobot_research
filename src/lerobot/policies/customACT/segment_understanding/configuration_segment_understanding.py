
from dataclasses import dataclass

@dataclass
class SegmentUnderstandingConfig:
    yolo_path: str = "yolo11l-seg.pt"