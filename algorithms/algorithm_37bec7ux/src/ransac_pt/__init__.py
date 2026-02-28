from dataclasses import dataclass

@dataclass
class Intrinsics:
    cx: float
    cy: float
    fx: float
    fy: float

@dataclass
class GridConfiguration:
    gw: float # grid width
    gh: float # grid height
    cw: float # cell width
    thres: int # points per cell to fill

@dataclass
class VirtualCamera:
    i: int
    j: int
    dir: float # radians
    fov: float # radians