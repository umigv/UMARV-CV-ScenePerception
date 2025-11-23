# ground plane mask to occupancy grid

from ransac import *

import ransac.plane

import numpy as np
import numpy.typing as npt

import math

# TODO: create a tool to tune grid paramters (scale, rotation, translation) in real time (or on a recording)


def create_point_cloud(ground_mask: npt.NDArray, depth_mask: npt.NDArray):
    c = np.argwhere(ground_mask != 1).astype(np.int64)
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
    # impact of cam plane [x, y, z] (inner elements) on real [x, y, z] (arrays)
    rotation_matrix = np.array([[1.0, 0.0, 0.0], [0.0, -s, c], [0.0, c, s]])

    return pixel_cloud @ rotation_matrix.transpose()  # reverse order because of format


def occupancy_grid(real_pc: npt.NDArray, conf: OccupancyGridShape):
    width = conf.gw // conf.cw
    height = conf.gh // conf.cw

    res = np.zeros((height, width), dtype=np.uint8)
    
    real_pc[:, 1] = real_pc[:, 2]

    real_pc = real_pc[:, :-1].astype(np.int16)
    real_pc[:, 0] = width / 2 + (real_pc[:, 0] // conf.cw)
    real_pc[:, 1] = height - 1 - (real_pc[:, 1] // conf.cw)

    real_pc[:, 0] = np.clip(real_pc[:, 0], 0, width - 1)
    real_pc[:, 1] = np.clip(real_pc[:, 1], 0, height - 1)

    threshold = 3 # TODO! threshold for density within occupancy grid cells
    # real_pc, pc_counts = np.unique(real_pc, axis=0, return_counts=True) very slow

    res[real_pc[:, 1], real_pc[:, 0]] = 1

    return res * 255
