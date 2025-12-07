from dataclasses import dataclass

@dataclass
class CameraIntrinsics:
    cx: float
    cy: float
    fx: float
    fy: float

@dataclass
class OccupancyGridConfiguration:
    gw: float # grid width
    gh: float # grid height
    cw: float # cell width
    thres: int # points per cell to fill