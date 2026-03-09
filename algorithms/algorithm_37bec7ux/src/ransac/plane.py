# ground plane mask creation

from ransac import *

import random
import math

from multiprocessing import Pool
import warnings
import numpy as np
import cv2


def pool(depths, kernel: tuple[int, int]):
    h, w = depths.shape
    w -= w % kernel[1]
    h -= h % kernel[0]
    warnings.simplefilter("ignore", category=RuntimeWarning)
    return np.nanmean(depths[:h, :w].reshape(h//kernel[0], kernel[0], w//kernel[1], kernel[1]), axis=(1, 3))


def sample(pooled):
    h, w = pooled.shape
    A = np.zeros((3, 3))
    b = np.zeros(3)

    while True:
        for i in range(3):
            row = -1
            col = -1
            while row < 0 or pooled[row][col] < 0:
                row = random.randint(0, h - 1)
                col = random.randint(0, w - 1)
            A[i] = [float(col), float(row), 1.0]
            b[i] = pooled[row][col]
        if np.linalg.matrix_rank(A) == 3:
            break
    return A, np.transpose(b)


def plane(A, b):
    return np.linalg.lstsq(A, b, rcond=None)[0]


def metric(depths, coeffs, tol: float):
    c1, c2, c3 = coeffs
    h, w = depths.shape

    x = np.arange(w, dtype=depths.dtype)[None, :]
    y = np.arange(h, dtype=depths.dtype)[:, None]

    r = c1 * x + c2 * y
    r += c3
    r -= depths
    np.abs(r, out=r)

    return np.count_nonzero((depths > 0) & (r < tol))


def mask(depths, coeffs, tol: float):
    h, w = depths.shape
    c1, c2, c3 = coeffs

    x = np.arange(w, dtype=depths.dtype)[None, :]
    y = np.arange(h, dtype=depths.dtype)[:, None]
    r = (c1 * x + c2 * y + c3) - depths
    return (depths > 0) & (r * r < tol)


def clean_depths(depths):
    depths = np.where(depths > 10000, np.nan, depths)
    return depths


def _ground_plane(pooled, tol, times):
    res = None
    best = 0
    for _ in range(times):
        coeffs = plane(*sample(pooled))
        score = metric(pooled, coeffs, tol)
        if score > best:
            res = coeffs
            best = score
    return best, res

# TODO: make this handle multiple depth frames
def ground_plane(
        depths, samples = 100, kernel = (1, 16), tol = 0.12, guess = np.array([0.0, 0.0, 0.0]), thread_pool = None, processes = 4):
    depths = clean_depths(depths)
    max_depth = float(np.nanmax(depths))
    if max_depth is math.nan:
        return np.zeros_like(depths), guess
    inv_depths = max_depth / depths

    pooled = pool(inv_depths, kernel)

    best_coeffs = guess.astype(float)
    if guess.shape != (3,):
        print("warning: invalid plane coefficient estimates")
        best_coeffs = np.array([0.0, 0.0, 0.0])
    else:
        best_coeffs *= float(max_depth)
        best_coeffs[0] *= float(kernel[1])
        best_coeffs[1] *= float(kernel[0])
    best = metric(pooled, best_coeffs, tol)

    if thread_pool is None:
        run_best, run_coeffs = _ground_plane(pooled, tol, samples)
        if run_best > best:
            best_coeffs = run_coeffs
    else:
        args = (pooled, tol, samples // processes)
        results = thread_pool.starmap(
            _ground_plane, [args for _ in range(processes)])
        _, best_coeffs = max(results, key=lambda t: t[0])

    best_coeffs[0] /= kernel[1]
    best_coeffs[1] /= kernel[0]

    res = mask(inv_depths, best_coeffs, tol)

    return res, np.array(best_coeffs) / max_depth


def hsv_mask(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 190], dtype=np.uint8)
    upper_white = np.array([255, 50, 255], dtype=np.uint8)
    return cv2.inRange(image, lower_white, upper_white)


def hsv_and_ransac(image, *args):
    ground_mask, coeffs = ground_plane(*args)
    lane_mask = hsv_mask(image) & ground_mask
    driveable = (ground_mask & (lane_mask != 1)).astype(np.uint8) * 255

    close_kernel = np.ones((2, 2), np.uint8)
    driveable = cv2.morphologyEx(driveable, cv2.MORPH_CLOSE, close_kernel)
    open_kernel = np.ones((7, 7), np.uint8)
    driveable = cv2.morphologyEx(driveable, cv2.MORPH_OPEN, open_kernel)
    return driveable.astype(bool), coeffs


def real_coeffs(best_coeffs, intrinsics: Intrinsics):
    c1, c2, c3 = best_coeffs
    # d = depth at the focal point
    d = 1 / (c1 * intrinsics.cx + c2 * intrinsics.cy + c3)
    return (-d * c1 * intrinsics.fx, d * c2 * intrinsics.fy, d)


# angle of depression
def real_angle(real_coeffs):
    a, b, _ = real_coeffs
    rad = math.acos(1 / math.hypot(a, b, 1))
    if (math.isnan(rad)):
        return 0
    return math.pi / 2 - rad
