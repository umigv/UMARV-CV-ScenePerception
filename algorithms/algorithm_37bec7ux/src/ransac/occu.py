# ground plane mask to occupancy grid

import ransac

import numpy as np
from numpy.typing import NDArray

# TODO: create a tool to tune grid paramters (scale, rotation, translation) in real time (or on a recording)

def create_point_cloud(ground_mask: NDArray):
    h, w = ground_mask.shape
    pass

def pixel_to_real(pixel_point_cloud, intrinsics: ransac.CameraIntrinsics):
    pass

def occupancy_grid(real_point_cloud: list[tuple[int, int]]):
    pass