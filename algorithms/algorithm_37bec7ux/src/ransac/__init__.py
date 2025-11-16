from dataclasses import dataclass


@dataclass
class CameraIntrinsics:
    cx: float
    cy: float
    fx: float
    fy: float
