import torch
import numpy as np
import cv2
import math
import skimage.draw


def get_device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def create_ground_cloud(coords, ransac_coeffs):
    device = get_device()

    coords = torch.as_tensor(coords, dtype=torch.float32, device=device)
    coeffs = torch.as_tensor(ransac_coeffs, dtype=torch.float32, device=device)

    c1, c2, c3 = coeffs
    z = 1.0 / (c1 * coords[:, 0] + c2 * coords[:, 1] + c3)
    z = z.unsqueeze(1)

    return torch.cat((coords, z), dim=1)


def create_point_cloud(mask, depth_map, skip: int = 3):
    device = get_device()

    mask = torch.as_tensor(mask, device=device)
    depth_map = torch.as_tensor(depth_map, device=device)

    coords = torch.nonzero(mask, as_tuple=False)
    coords = coords[:, [1, 0]].float()

    depths = depth_map[coords[:, 1].long(), coords[:, 0].long()].unsqueeze(1)

    res = torch.cat((coords, depths), dim=1)

    return res[::1 + skip]


def pixel_to_real(pixel_cloud, real_coeffs, intr, orientation=0.0):
    device = get_device()

    cloud = pixel_cloud.clone().to(device)
    real_coeffs = torch.as_tensor(real_coeffs, device=device)

    cloud[:, 0] = cloud[:, 2] * (cloud[:, 0] - intr.cx) / intr.fx
    cloud[:, 1] = cloud[:, 2] * (intr.cy - cloud[:, 1]) / intr.fy

    a, b, _ = real_coeffs
    depression = math.acos(1 / math.sqrt(a*a + b*b + 1))
    depression = math.pi/2 - depression

    c1 = math.cos(depression)
    s1 = math.sin(depression)

    Rx = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, c1, -s1],
        [0.0, s1, c1]
    ], device=device)

    c2 = math.cos(orientation)
    s2 = math.sin(orientation)

    Ry = torch.tensor([
        [c2, 0.0, -s2],
        [0.0, 1.0, 0.0],
        [s2, 0.0, c2]
    ], device=device)

    R = Rx @ Ry

    return cloud @ R.T


def occupancy_grid(real_pc, conf):
    device = get_device()
    real_pc = real_pc.to(device)

    width = conf.gw // conf.cw
    height = conf.gh // conf.cw

    pts = real_pc[:, (0, 2)].clone()

    pts[:, 0] = width // 2 + (pts[:, 0] // conf.cw)
    pts[:, 1] = height - 1 - (pts[:, 1] // conf.cw)

    pts = pts.long()

    valid = (
        (pts[:, 0] >= 0) & (pts[:, 0] < width) &
        (pts[:, 1] >= 0) & (pts[:, 1] < height)
    )
    pts = pts[valid]

    idx = pts[:, 1] * width + pts[:, 0]

    cnt = torch.bincount(idx, minlength=width*height)
    cnt = cnt.reshape(height, width)

    grid = cnt >= conf.thres

    return grid


def composite(drive_occ, block_occ):
    if isinstance(drive_occ, torch.Tensor):
        drive_occ = drive_occ.detach().cpu().numpy()
    if isinstance(block_occ, torch.Tensor):
        block_occ = block_occ.detach().cpu().numpy()

    full = drive_occ & (block_occ != 1)
    full = full.astype(np.uint8) * 255
    full[(block_occ | drive_occ) != 1] = 127
    return full


def constrain(points, w: int, h: int):
    if isinstance(points, torch.Tensor):
        pts = points
        valid = (
            (pts[:, 0] >= 0) & (pts[:, 0] < w) &
            (pts[:, 1] >= 0) & (pts[:, 1] < h)
        )
        return pts[valid]
    else:
        points = points.astype(int)
        valid = (
            (points[:, 0] >= 0) & (points[:, 0] < w) &
            (points[:, 1] >= 0) & (points[:, 1] < h)
        )
        return points[valid]


def fast_los_grid(merged, iters=10):
    if isinstance(merged, torch.Tensor):
        merged = merged.detach().cpu().numpy()

    merged = merged.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize=(2, 2))
    work = merged.copy()

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


def create_los_grid(merged, cameras=[]):
    if isinstance(merged, torch.Tensor):
        merged = merged.detach().cpu().numpy()

    merged = merged.astype(np.uint8)
    h, w = merged.shape

    if len(cameras) == 0:
        return fast_los_grid(merged)

    for cam in cameras:

        dx0 = math.cos(cam.dir - cam.fov / 2)
        dy0 = -math.sin(cam.dir - cam.fov / 2)
        dx1 = math.cos(cam.dir + cam.fov / 2)
        dy1 = -math.sin(cam.dir + cam.fov / 2)

        r = 2 * (h + w)
        x0, y0 = cam.j + int(dx0 * r), cam.i + int(dy0 * r)
        x1, y1 = cam.j + int(dx1 * r), cam.i + int(dy1 * r)

        nx0, nx1 = np.clip((x0, x1), 0, w - 1)
        y0 += (nx0 - x0) * dy0 / dx0 if dx0 != 0 else 0
        x0 = nx0
        y1 += (nx1 - x1) * dy1 / dx1 if dx1 != 0 else 0
        x1 = nx1

        ny0, ny1 = np.clip((y0, y1), 0, h - 1)
        x0 += (ny0 - y0) * dx0 / dy0 if dy0 != 0 else 0
        y0 = ny0
        x1 += (ny1 - y1) * dx1 / dy1 if dy1 != 0 else 0
        y1 = ny1

        x0, x1 = np.clip((x0, x1), 0, w - 1)
        y0, y1 = np.clip((y0, y1), 0, h - 1)
        x0, x1, y0, y1 = int(x0), int(x1), int(y0), int(y1)

        idx, jdx = [], []

        while x0 != x1 or y0 != y1:
            idx.append(int(np.clip(y0, 0, h - 1)))
            jdx.append(int(np.clip(x0, 0, w - 1)))

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
        
        idx = idx[0:len(idx)//2:-1] + idx[len(idx)//2:len(idx)-1]
        jdx = jdx[0:len(jdx)//2:-1] + jdx[len(jdx)//2:len(jdx)-1]

        for end_i, end_j in zip(idx, jdx):
            state = 255
            line = skimage.draw.line(cam.i, cam.j, end_i, end_j)

            for p in range(len(line[0])):
                val = merged[line[0][p], line[1][p]]
                if val == 0:
                    state = 0
                elif val == 255:
                    state = 255
                else:
                    merged[line[0][p], line[1][p]] = state

    return merged
