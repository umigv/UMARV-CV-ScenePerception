import torch
import torch.nn.functional as F
import math
import random
import cv2


def get_device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def clean_depths(depths):
    device = get_device()
    depths = torch.as_tensor(depths, dtype=torch.float32, device=device)
    depths = torch.where(torch.isinf(depths) |
                         torch.isnan(depths), -1.0, depths)
    depths = torch.clamp(depths, max=10000.0)
    return depths


def pool(depths, kernel: tuple[int, int]):
    device = get_device()
    depths = depths.to(device)

    h, w = depths.shape
    kh, kw = kernel
    h -= h % kh
    w -= w % kw
    depths = depths[:h, :w]
    depths = depths.unsqueeze(0).unsqueeze(0)
    pooled = F.max_pool2d(depths, kernel_size=kernel)
    return pooled.squeeze(0).squeeze(0)


def sample(pooled, batch: int, max_attempts: int = 10):
    device = pooled.device
    h, w = pooled.shape

    # Allocate outputs
    A_out = torch.empty((batch, 3, 3), device=device)
    b_out = torch.empty((batch, 3), device=device)

    remaining = torch.arange(batch, device=device)

    attempts = 0

    while remaining.numel() > 0 and attempts < max_attempts:
        n = remaining.numel()

        rows = torch.randint(0, h, (n, 3), device=device)
        cols = torch.randint(0, w, (n, 3), device=device)

        vals = pooled[rows, cols]

        valid_depth = (vals > 0).all(dim=1)

        ones = torch.ones((n, 3), device=device)

        A = torch.stack([
            cols.float(),
            rows.float(),
            ones
        ], dim=-1)

        det = torch.linalg.det(A)
        valid_rank = det.abs() > 1e-6

        valid = valid_depth & valid_rank

        if valid.any():
            idx_valid = remaining[valid]
            A_out[idx_valid] = A[valid]
            b_out[idx_valid] = vals[valid].float()

        remaining = remaining[~valid]
        attempts += 1

    # ---------- FAILSAFE ----------
    if remaining.numel() > 0:
        # Deterministic fallback:
        # pick first 3 valid depth points in image

        valid_mask = pooled > 0
        ys, xs = torch.nonzero(valid_mask, as_tuple=True)

        if ys.numel() >= 3:
            rows = ys[:3].unsqueeze(0).repeat(remaining.numel(), 1)
            cols = xs[:3].unsqueeze(0).repeat(remaining.numel(), 1)
            vals = pooled[rows, cols]

            ones = torch.ones((remaining.numel(), 3), device=device)

            A = torch.stack([
                cols.float(),
                rows.float(),
                ones
            ], dim=-1)

            A_out[remaining] = A
            b_out[remaining] = vals.float()
        else:
            # extreme case: almost no valid depth
            A_out[remaining] = torch.eye(3, device=device)
            b_out[remaining] = torch.ones(
                (remaining.numel(), 3), device=device)

    return A_out, b_out


def plane(A, b, eps: float = 1e-6):
    # Regularize slightly to avoid singular matrix crashes
    I = torch.eye(3, device=A.device)
    return torch.linalg.solve(A + eps * I, b.unsqueeze(-1)).squeeze(-1)


def metric(pooled, coeffs, tol: float):

    device = pooled.device
    pooled = pooled.to(device)

    if coeffs.ndim == 1:
        coeffs = coeffs.unsqueeze(0)
        squeeze_out = True
    else:
        squeeze_out = False

    N = coeffs.shape[0]
    h, w = pooled.shape

    ys, xs = torch.meshgrid(
        torch.arange(h, device=device),
        torch.arange(w, device=device),
        indexing="ij"
    )

    c = coeffs.view(N, 1, 1, 3)

    z_pred = c[..., 0] * xs + c[..., 1] * ys + c[..., 2]

    err = torch.abs(z_pred - pooled)

    valid = (pooled > 0)
    inliers = valid & (err < tol)

    scores = inliers.sum(dim=(1, 2))

    if squeeze_out:
        return scores[0]

    return scores


def mask(depths, coeffs, tol: float):
    device = get_device()
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
    guess=None
):
    device = depths.device

    depths = clean_depths(depths)
    max_depth = depths.max()
    inv_depths = max_depth / depths

    pooled = pool(inv_depths, kernel)

    A, b = sample(pooled, batch=iters)

    coeffs = plane(A, b)

    scores = metric(pooled, coeffs, tol)

    if guess is not None:
        guess = torch.as_tensor(guess, dtype=torch.float32, device=device)
        guess_score = metric(pooled, guess, tol)
        coeffs = torch.cat([coeffs, guess.unsqueeze(0)], dim=0)
        scores = torch.cat([scores, guess_score.unsqueeze(0)], dim=0)

    best_idx = torch.argmax(scores)
    best_coeffs = coeffs[best_idx]

    best_coeffs = best_coeffs.clone()
    best_coeffs[0] /= kernel[1]
    best_coeffs[1] /= kernel[0]

    res = mask(inv_depths, best_coeffs, tol)

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
