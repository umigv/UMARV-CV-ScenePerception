from dataclasses import dataclass


@dataclass
class Intrinsics:
    cx: float
    cy: float
    fx: float
    fy: float


@dataclass
class GridConfiguration:
    gw: float  # grid width in mm
    gh: float  # grid height in mm
    cw: float  # cell width in mm
    thres: int = 1  # points per cell to fill


@dataclass
class VirtualCamera:
    i: int
    j: int
    dir: float  # radians
    fov: float  # radians
