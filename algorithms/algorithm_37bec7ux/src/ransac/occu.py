# ground plane mask to occupancy grid

from ransac import *

import ransac.plane

import numpy as np
import numpy.typing as npt

import math

# TODO: create a tool to tune grid paramters (scale, rotation, translation) in real time (or on a recording)


def create_point_cloud(ground_mask: npt.NDArray, depth_mask: npt.NDArray):
    c = np.argwhere(ground_mask).astype(np.int64)
    c[:, [0, 1]] = c[:, [1, 0]]  # swap rows and cols indices
    depths = depth_mask[c[:, 1], c[:, 0]].reshape(-1, 1)  # shorthand for speed
    return np.concatenate((c.astype(np.float64), depths), axis=1)


def pixel_to_real(
    pixel_cloud: npt.NDArray, real_coeffs: npt.NDArray, intr: CameraIntrinsics
):

    pixel_cloud[:, 0] = pixel_cloud[:, 2] * (pixel_cloud[:, 0] - intr.cx) / intr.fx
    pixel_cloud[:, 1] = pixel_cloud[:, 2] * (intr.cy - pixel_cloud[:, 1]) / intr.fy

    angle = ransac.plane.real_angle(real_coeffs)
    c = math.cos(angle)
    s = math.sin(angle)
    rotation_matrix = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, c, s]])

    return pixel_cloud @ rotation_matrix.transpose()  # reverse order because of format


def occupancy_grid(real_cloud: npt.NDArray):
    pass
    grouping = 50
    real_cloud.astype(np.int64)
