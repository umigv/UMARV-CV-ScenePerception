# ground plane mask to occupancy grid

from ransac import *

import ransac.plane

import numpy as np
import numpy.typing as npt
import cv2
import skimage

import math

# TODO: create a tool to tune grid paramters (scale, rotation, translation) in real time (or on a recording)


def create_point_cloud(mask: npt.NDArray, depth_mask: npt.NDArray):
    coords = np.argwhere(mask).astype(np.int64)
    coords[:, [0, 1]] = coords[:, [1, 0]]  # swap rows and cols indices
    depths = depth_mask[coords[:, 1], coords[:, 0]].reshape(-1, 1)

    return np.concatenate((coords.astype(np.float64), depths), axis=1)

# outputs (x,y,z) with real z as depth, y as height
# y value outputs are complete garbage
def pixel_to_real(
    pixel_cloud: npt.NDArray, real_coeffs: npt.NDArray, intr: CameraIntrinsics
):
    # converts px into mm
    pixel_cloud[:, 0] = pixel_cloud[:, 2] * (pixel_cloud[:, 0] - intr.cx) / intr.fx
    pixel_cloud[:, 1] = pixel_cloud[:, 2] * (intr.cy - pixel_cloud[:, 1]) / intr.fy

    angle = ransac.plane.real_angle(real_coeffs)
    c = math.cos(angle)
    s = math.sin(angle)
    # cosine, sine reversed from usual because of angle output
    # impact of cam plane [x, y, z] (inner elements) on real [x, y, z] (arrays)
    rotation_matrix = np.array([[1.0, 0.0, 0.0],
                                [0.0,   s,   c],
                                [0.0,  -c,   s]])

    return pixel_cloud @ rotation_matrix # reverse order because of format


def bind_idx(points: npt.NDArray, w: int, h: int):
    points = points.astype(int)
    valid = (points[:, 1] >= 0) & (points[:, 1] < w)
    valid &= (points[:, 0] >= 0) & (points[:, 0] < h)
    return points[valid]


def occupancy_grid(real_pc: npt.NDArray, conf: OccupancyGridConfiguration):
    width = conf.gw // conf.cw
    height = conf.gh // conf.cw

    real_pc = real_pc[:, (0, 2)]

    real_pc = real_pc.astype(np.int16)
    real_pc[:, 0] = width // 2 + (real_pc[:, 0] // conf.cw)
    real_pc[:, 1] = height - 1 - (real_pc[:, 1] // conf.cw)
    real_pc = bind_idx(real_pc, height, width)  # reversed order because x, y not y, x

    cnt = np.bincount(real_pc[:, 1] * width + real_pc[:, 0])
    cnt = np.resize(cnt, (height, width))

    grid = cnt >= conf.thres

    return grid


def merge(drive_occ: npt.NDArray, block_occ: npt.NDArray):
    merged = drive_occ & (block_occ != 1)
    merged = merged.astype(np.uint8) * 255
    unknown = block_occ | drive_occ != 1
    merged[unknown] = 127
    return merged


def fast_los_grid(merged: npt.NDArray, iters=10):
    merged = merged.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize=(2, 2))
    work = merged
    for i in range(iters):
        work = cv2.erode(work, kernel, iterations=2 * i)
        merged[(merged == 127) & (work == 255)] = 255
        work[merged == 255] = 255
        work[merged == 0] = 0
        work = cv2.dilate(work, kernel, iterations=i)
        merged[(merged == 127) & (work == 0)] = 0
        work[merged == 255] = 255
        work[merged == 0] = 0
    return work


def create_los_grid(merged: npt.NDArray, cameras: list[VirtualCamera] = []):
    # merged: 2-d boolean array with 0/255 as known driveable/undriveable
    #         all other values are unknown
    merged = merged.astype(np.uint8)
    h, w = merged.shape

    if len(cameras) == 0:
        return fast_los_grid(merged)

    for cam in cameras:
        # scan right to left
        dx0 = math.cos(cam.dir - cam.fov / 2)
        dy0 = -math.sin(cam.dir - cam.fov / 2)
        dx1 = math.cos(cam.dir + cam.fov / 2)
        dy1 = -math.sin(cam.dir + cam.fov / 2)

        r = 2 * (h + w)
        x0, y0 = cam.j + int(dx0 * r), cam.i + int(dy0 * r)
        x1, y1 = cam.j + int(dx1 * r), cam.i + int(dy1 * r)

        # restrict x
        nx0, nx1 = np.clip((x0, x1), 0, w - 1)
        y0 += (nx0 - x0) * dy0 / dx0
        x0 = nx0
        y1 += (nx1 - x1) * dy1 / dx1
        x1 = nx1

        # restrict y
        ny0, ny1 = np.clip((y0, y1), 0, h - 1)
        x0 += (ny0 - y0) * dx0 / dy0
        y0 = ny0
        x1 += (ny1 - y1) * dx1 / dy1
        y1 = ny1

        x0, x1 = np.clip((x0, x1), 0, w - 1)
        y0, y1 = np.clip((y0, y1), 0, h - 1)
        x0, x1, y0, y1 = int(x0), int(x1), int(y0), int(y1)

        idx, jdx = [], []
        while x0 != x1 or y0 != y1:
            idx.append(int(np.clip(y0, 0, h - 1)))
            jdx.append(int(np.clip(x0, 0, w - 1)))
            # traverse along image boundary acw
            if x0 == 0 and y0 < h:
                y0 += 1
            elif y0 == h - 1 and x0 < w:
                x0 += 1
            elif x0 == w - 1 and y0 > 0:
                y0 -= 1
            elif y0 == 0 and x0 > 0:
                x0 -= 1
            else:
                break

        merged[cam.i, cam.j] = 255
        for end_i, end_j in zip(idx, jdx):
            state = 255
            line = skimage.draw.line(
                cam.i, cam.j, end_i, end_j
            )  # FIXME: bottleneck is this function
            for p in range(len(line[0])):
                if merged[line[0][p], line[1][p]] == 0:
                    state = 0
                elif merged[line[0][p], line[1][p]] == 255:
                    state = 255
                else:
                    merged[line[0][p], line[1][p]] = state

    return merged
