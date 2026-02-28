import torch
import torch.nn.functional as F
import math
import random
import cv2


def _device(cuda: bool):
    return torch.device("cuda") if cuda and torch.cuda.is_available() else torch.device("cpu")


def clean_depths(depths, cuda: bool = False):
    device = _device(cuda)
    depths = torch.as_tensor(depths, dtype=torch.float32, device=device)
    depths = torch.where(torch.isinf(depths) | torch.isnan(depths), -1.0, depths)
    depths = torch.clamp(depths, max=10000.0)
    return depths


def pool(depths, kernel: tuple[int, int], cuda: bool = False):
    device = _device(cuda)
    depths = depths.to(device)

    h, w = depths.shape
    kh, kw = kernel
    h -= h % kh
    w -= w % kw
    depths = depths[:h, :w]
    depths = depths.unsqueeze(0).unsqueeze(0)
    pooled = F.max_pool2d(depths, kernel_size=kernel)
    return pooled.squeeze(0).squeeze(0)


def sample(pooled, cuda: bool = False):
    device = _device(cuda)
    h, w = pooled.shape

    while True:
        rows = torch.randint(0, h, (3,), device=device)
        cols = torch.randint(0, w, (3,), device=device)

        valid = pooled[rows, cols] > 0
        if valid.sum() < 3:
            continue

        A = torch.stack([
            torch.stack([cols[i].float(), rows[i].float(), torch.tensor(1.0, device=device)])
            for i in range(3)
        ])
        if torch.linalg.matrix_rank(A) == 3:
            b = pooled[rows, cols].float()
            return A, b


def plane(A, b, cuda: bool = False):
    return torch.linalg.lstsq(A, b).solution


def metric(pooled, coeffs, tol: float, cuda: bool = False):
    device = _device(cuda)
    pooled = pooled.to(device)

    c1, c2, c3 = coeffs
    h, w = pooled.shape

    ys, xs = torch.meshgrid(
        torch.arange(h, device=device),
        torch.arange(w, device=device),
        indexing="ij"
    )

    z_pred = c1 * xs + c2 * ys + c3
    err = torch.abs(z_pred - pooled)

    return torch.count_nonzero((pooled > 0) & (err < tol))


def mask(depths, coeffs, tol: float, cuda: bool = False):
    device = _device(cuda)
    depths = depths.to(device)

    h, w = depths.shape
    X, Y = torch.meshgrid(
        torch.arange(w, device=device),
        torch.arange(h, device=device),
        indexing="xy"
    )

    c1, c2, c3 = coeffs
    Z = (c1 * X + c2 * Y + c3 - depths) ** 2

    return (depths > 0) & (Z < tol)


def ground_plane(
    depths,
    iters: int = 60,
    kernel: tuple[int, int] = (1, 16),
    tol: float = 0.12,
    guess=None,
    cuda: bool = False
):
    device = _device(cuda)

    depths = clean_depths(depths, cuda)
    max_depth = depths.max()
    inv_depths = max_depth / depths

    pooled = pool(inv_depths, kernel, cuda)

    if guess is None:
        best_coeffs = torch.zeros(3, device=device)
    else:
        best_coeffs = torch.as_tensor(guess, dtype=torch.float32, device=device)

    best = metric(pooled, best_coeffs, tol, cuda)

    for _ in range(iters):
        A, b = sample(pooled, cuda)
        coeffs = plane(A, b, cuda)
        score = metric(pooled, coeffs, tol, cuda)

        if score > best:
            best = score
            best_coeffs = coeffs

    best_coeffs[0] /= kernel[1]
    best_coeffs[1] /= kernel[0]

    res = mask(inv_depths, best_coeffs, tol, cuda)

    return res, best_coeffs / max_depth


def real_coeffs(best_coeffs, intrinsics):
    if isinstance(best_coeffs, torch.Tensor):
        coeffs = best_coeffs.detach().cpu().numpy()
    else:
        coeffs = best_coeffs

    c1, c2, c3 = coeffs

    d = 1.0 / (c1 * intrinsics.cx + c2 * intrinsics.cy + c3)

    return (
        -d * c1 * intrinsics.fx,
         d * c2 * intrinsics.fy,
         d
    )


def real_angle(real_coeffs):
    a, b, _ = real_coeffs

    denom = math.sqrt(a * a + b * b + 1.0)
    if denom == 0:
        return 0.0

    val = 1.0 / denom

    val = max(-1.0, min(1.0, val))

    rad = math.acos(val)

    if math.isnan(rad):
        return 0.0

    return math.pi / 2 - rad