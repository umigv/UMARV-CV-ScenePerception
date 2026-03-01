# ground plane mask to occupancy grid

from numba import njit
from ransac import *

import ransac.plane

import numpy as np
import cv2

import math


def create_ground_cloud(coords, ransac_coeffs):
    # coords is a Nx2 numpy array containing coordinates (x, y)
    # pass pixel coefficients

    c1, c2, c3 = ransac_coeffs

    z = 1 / (c1 * coords[:, 0] + c2 * coords[:, 1] + c3)
    z = z.reshape(-1, 1)
    return np.concatenate((coords.astype(np.float64), z), axis=1)


def create_point_cloud(mask, depth_map, skip: int = 3):
    coords = np.argwhere(mask).astype(np.int64)
    coords[:, [0, 1]] = coords[:, [1, 0]]  # (row, col) -> (x, y)
    depths = depth_map[coords[:, 1], coords[:, 0]].reshape(-1, 1)

    res = np.concatenate((coords.astype(np.float64), depths), axis=1)
    return res[0::1+skip]


def pixel_to_real(
        pixel_cloud, real_coeffs, intr: Intrinsics, orientation: float = 0.0):
    # outputs (x,y,z) with real z as depth, y as height
    # y values are relative to the camera's height
    # orientation (radians) is positive to orient the camera left

    # converts px into mm
    cloud = pixel_cloud.copy()
    cloud[:, 0] = pixel_cloud[:, 2] * (pixel_cloud[:, 0] - intr.cx) / intr.fx
    cloud[:, 1] = pixel_cloud[:, 2] * (intr.cy - pixel_cloud[:, 1]) / intr.fy

    depression = ransac.plane.real_angle(real_coeffs)
    c_1 = math.cos(depression)
    s_1 = math.sin(depression)
    # each column affects the output (x, y, z) respectively
    rotation_matrix = np.array([[1.0, 0.0,  0.0],
                                [0.0, c_1, -s_1],
                                [0.0, s_1,  c_1]]).transpose()

    c_2 = math.cos(orientation)
    s_2 = math.sin(orientation)
    rotation_matrix = rotation_matrix @ np.array([[c_2, 0.0, -s_2],
                                                  [0.0, 1.0,  0.0],
                                                  [s_2, 0.0,  c_2]]).transpose()

    return cloud @ rotation_matrix


def constrain(points, w: int, h: int):
    points = points.astype(int)
    valid = (points[:, 0] >= 0) & (points[:, 0] < w) & (
        points[:, 1] >= 0) & (points[:, 1] < h)
    return points[valid]


def occupancy_grid(real_pc, conf: GridConfiguration):
    width = conf.gw // conf.cw
    height = conf.gh // conf.cw

    real_pc = real_pc[:, (0, 2)]

    real_pc = real_pc.astype(np.int16)
    real_pc[:, 0] = width // 2 + (real_pc[:, 0] // conf.cw)
    real_pc[:, 1] = height - 1 - (real_pc[:, 1] // conf.cw)
    real_pc = constrain(real_pc, width, height)

    cnt = np.bincount(real_pc[:, 1] * width + real_pc[:, 0])
    cnt = np.resize(cnt, (height, width))

    return cnt >= conf.thres


def composite(drive_occ, block_occ):
    full = drive_occ & (block_occ != 1)
    full = full.astype(np.uint8) * 255
    full[(block_occ | drive_occ) != 1] = 127
    return full


def fast_los_grid(merged, iters=10):
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


@njit(cache=True)
def trace_and_fill(merged, i0, j0, i1, j1):
    # Bresenham walk from (i0,j0) -> (i1,j1)
    di = abs(i1 - i0)
    dj = abs(j1 - j0)
    si = 1 if i0 < i1 else -1
    sj = 1 if j0 < j1 else -1
    err = di - dj

    i, j = i0, j0
    state = 255

    while True:
        v = merged[i, j]
        if v == 0:
            state = 0
        elif v == 255:
            state = 255
        else:
            merged[i, j] = state

        if i == i1 and j == j1:
            break

        e2 = err + err
        if e2 > -dj:
            err -= dj
            i += si
        if e2 < di:
            err += di
            j += sj


def create_los_grid(merged, cameras: list[VirtualCamera] = []):
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
        nx0 = np.clip(x0, 0, w-1)
        nx1 = np.clip(x1, 0, w-1)
        y0 += (nx0 - x0) * dy0 / dx0
        x0 = nx0
        y1 += (nx1 - x1) * dy1 / dx1
        x1 = nx1

        # restrict y
        ny0 = np.clip(y0, 0, h-1)
        ny1 = np.clip(y1, 0, h-1)
        x0 += (ny0 - y0) * dx0 / dy0
        y0 = ny0
        x1 += (ny1 - y1) * dx1 / dy1
        y1 = ny1

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
            trace_and_fill(merged, cam.i, cam.j, end_i, end_j)

    return merged


# TODO decompose pitch + roll angles
# the numpy fuckery in this just helps interpolation
def oneshot(mask, real_coeffs, intr: Intrinsics, conf: GridConfiguration,
            h=0, ignore=None):
    # grid should be symmetric
    # first and second indices are number of layers to compute
    grid_shape = (3, 3, 2 * int((0.5 * conf.gh) // conf.cw),
                  2 * int((0.5 * conf.gw) // conf.cw))
    true_width = conf.cw * grid_shape[3]
    true_height = conf.cw * grid_shape[2]

    grid = np.zeros(grid_shape, dtype=np.uint8)

    # go there, is the difference in depth of the prediction matching the depth at that actual place? is this process the same as the masking process? yes

    lys = np.arange(grid_shape[0])[:, None, None, None]
    lxs = np.arange(grid_shape[1])[None, :, None, None]
    gys = np.arange(grid_shape[2])[None, None, :, None]
    gxs = np.arange(grid_shape[3])[None, None, None, :]

    # apply camera rotation
    rgys = grid_shape[2] - gys - 0.5
    rgxs = gxs - grid_shape[3] / 2 + 0.5
    rgxs_temp = rgxs * math.cos(h) + rgys * math.sin(h)
    rgys_temp = -rgxs * math.sin(h) + rgys * math.cos(h)
    rgys = grid_shape[2] - rgys_temp - 0.5
    rgxs = rgxs_temp + grid_shape[3] / 2 - 0.5

    # pixel values into mm
    cxs = conf.cw * ((lxs + 0.5) / grid_shape[0] + rgxs) - 0.5 * true_width
    cys = true_height - conf.cw * (2 * (lys + 0.5) / grid_shape[1] + rgys)

    # project onto the camera plane
    a, b, d = real_coeffs

    theta = ransac.plane.real_angle(real_coeffs)
    cam_height = math.sin(theta) * d
    cys = cys * math.sin(theta)
    cys = cys - math.cos(theta) * cam_height

    # use mask to highlight driveable regions
    # python matrix nonsense that somehow works

    zs = a * cxs + b * cys + d
    pxs = np.round((cxs * intr.fx) / zs + intr.cx)
    pys = np.round(intr.cy - (cys * intr.fy) / zs)

    pxs = np.clip(pxs, 0, mask.shape[1] - 1).astype(np.int16)
    pys = np.clip(pys, 0, mask.shape[0] - 1).astype(np.int16)
    grid[lys, lxs, gys, gxs] = mask[pys, pxs]
    grid = (255 * (np.mean(grid, axis=(0, 1)) >= 0.75)).astype(np.uint8)
    if ignore is not None:
        grid[ignore[0]:ignore[2], ignore[1]: ignore[3]] = 127

    return grid
