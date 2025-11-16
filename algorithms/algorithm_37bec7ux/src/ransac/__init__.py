import random
import math

import numpy as np
import skimage
import cv2


def pool(depths, kernel: tuple[int, int]):
    h, w = depths.shape
    w -= w % kernel[1]
    h -= h % kernel[0]
    depths = depths[:h, :w]
    return skimage.measure.block_reduce(depths, kernel, np.mean)


def sample(pooled):
    h, w = pooled.shape
    A = np.zeros((3, 3))
    b = np.zeros(3)
    while True:
        for i in range(3):
            row = -1
            col = -1
            while row < 0 or np.isneginf(pooled[row][col]):
                row = random.randint(0, h - 1)
                col = random.randint(0, w - 1)
            A[i] = [float(col), float(row), 1.0]
            b[i] = pooled[row][col]
        if np.linalg.matrix_rank(A) == 3:
            break
    return A, np.transpose(b)


def plane(A, b):
    return np.linalg.lstsq(A, b, rcond=None)[0]


def metric(pooled, coeffs, tol: float):
    c1, c2, c3 = coeffs
    h, w = pooled.shape
    ys, xs = np.indices((h, w))
    z_pred = c1 * xs + c2 * ys + c3
    err = np.abs(z_pred - pooled)
    return np.count_nonzero(err < tol)


def mask(depths, coeffs, tol: float):
    h, w = depths.shape
    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    c1, c2, c3 = coeffs
    Z = pow(c1 * X + c2 * Y + c3 - depths, 2)
    return Z < tol


def clean_depths(depths):
    depths = np.where(np.isinf(depths) | np.isnan(depths), -np.inf, depths)
    return depths


# will maintain the dimensions of the original
def ransac(
    depths, iters: int = 60, kernel: tuple[int, int] = (1, 12), tol: float = 0.1
):
    max_depth = float(depths.max())
    depths = max_depth / depths
    pooled = pool(depths, kernel)
    best = 0
    best_coeffs = [0, 0, 0]

    for _ in range(iters):
        A, b = sample(pooled)
        coeffs = plane(A, b)
        score = metric(pooled, coeffs, tol)
        if score > best:
            best = score
            best_coeffs = coeffs

    best_coeffs[0] /= kernel[1]
    best_coeffs[1] /= kernel[0]

    res = mask(depths, best_coeffs, tol)

    return res, np.array(best_coeffs) / max_depth


def hsv_mask(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 180], dtype=np.uint8)
    upper_white = np.array([180, 50, 255], dtype=np.uint8)
    white_mask = cv2.inRange(image, lower_white, upper_white) > 0

    return white_mask


def hsv_and_ransac(image, *args):
    ground_mask, coeffs = ransac(*args)
    lane_mask = hsv_mask(image) & ground_mask
    outlier_mask = ground_mask != 1

    return outlier_mask | lane_mask, coeffs


def real_coeffs(best_coeffs, cx, cy, fx, fy):
    c1, c2, c3 = best_coeffs
    d = 1 / (c1 * cx + c2 * cy + c3)
    a = -d * c1 * fx
    b = -d * c2 * fy
    return a, b, d


def real_angle(real_coeffs):
    # upon clicking on the opencv image
    # get the pixel coordinates of the click
    # angle between [0, 0, -1] and [a, b, -1]
    a, b, _ = real_coeffs
    return math.acos(1 / math.hypot(a, b, -1))
