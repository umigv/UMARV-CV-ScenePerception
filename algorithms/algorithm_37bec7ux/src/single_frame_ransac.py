import numpy as np  # matrices
import matplotlib.pyplot as plt
import h5py  # reading data
import random
import time
import math
import cv2
import cProfile as cp
import pstats
from multiprocessing import Pool
import io

from ransac import *
from ransac import plane, occu

# PARAMETERS

filename = "res/perspective_test.svo2.hdf5"
frame_number = -1

samples = 100
kernel = (1, 16)  # kernel is rows, columns
tolerance = 0.1
processes = 8

# INPUT FILTERING

f = h5py.File(filename, "r")
frames = len(f["depth_maps"])

if frame_number < 0:
    frame_number = random.randint(1, frames - 2)
elif frame_number >= frames:
    frame_number = frames - 1

raw_depths = f["depth_maps"][frame_number]
depth_map = f["depth_maps"][frame_number]
image = f["images"][frame_number]
image = image[:, 0: int(image.shape[1] / 2)]

print()
print("testing on:", filename)
print(".hdf5 keys:", list(f.keys()))
print(f"using frame number: {frame_number}")
print("\n----- start -----\n")

f.close()

pool = Pool(processes) if processes > 0 else None

# START
pr = cp.Profile()
pr.enable()
start = time.perf_counter_ns()

depths = plane.clean_depths(raw_depths)
driveable, ransac_coeffs = plane.hsv_and_ransac(
    image, depths, samples, kernel, tolerance, np.array([0, 0, 0]), pool, processes)

ransac_output = driveable  # [100:, :]

fx = 360

h, w = depth_map.shape
intrinsics = Intrinsics(w / 2, h / 2, fx, fx)
real = plane.real_coeffs(ransac_coeffs, intrinsics)
angle = plane.real_angle(real)

# drive_ppc = ransac.occu.create_point_cloud(driveable, cleaned_depths)
# drive_rpc = ransac.occu.pixel_to_real(drive_ppc, real, intrinsics, math.pi/4)

# block_ppc = ransac.occu.create_point_cloud(driveable != 1, cleaned_depths)
# block_rpc = ransac.occu.pixel_to_real(block_ppc, real, intrinsics, math.pi/4)

# drive_conf = ransac.GridConfiguration(
#     5000, 5000, 50, thres=2)  # in millimetres
# block_conf = ransac.GridConfiguration(
#     5000, 5000, 50, thres=1)  # in millimetres
# drive_occ = ransac.occu.occupancy_grid(drive_rpc, drive_conf)
# block_occ = ransac.occu.occupancy_grid(block_rpc, block_conf)
# full_occ = ransac.occu.composite(drive_occ, block_occ)

# occ_h, occ_w = full_occ.shape
# cam = ransac.VirtualCamera(occ_h - 1, occ_w // 2,
#                            3 * math.pi / 4, math.radians(90))

# TEST NEW OCCUPANCY GRID
# start = time.perf_counter_ns()
conf = GridConfiguration(5000, 5000, 50)
occ = occu.oneshot(ransac_output, real, intrinsics, conf, math.pi / 4)

end = time.perf_counter_ns()
pr.disable()

# IMPORTANT don't profile this, has JIT compile time of 0.5s
# los_grid = ransac.occu.create_los_grid(full_occ, [cam])

s = io.StringIO()
pstats.Stats(pr, stream=s).strip_dirs().sort_stats("tottime").print_stats(20)
# print(s.getvalue())
pr.dump_stats("out/single_frame.prof")

# DISPLAY DATA

print(f"\n----- {(end - start) / 1e6:.0f} ms -----\n")

print("coeffs: ", ransac_coeffs)
print("angle: ", math.degrees(angle))

# exit()

# PLOT THINGS


def show_pc(axes, cloud, conf: GridConfiguration, name: str = "point cloud"):
    axes.set_title(name)
    axes.scatter(cloud[:, 0], cloud[:, 2], s=0.01)
    axes.set_aspect("equal", adjustable="box")
    axes.set_xlim((-conf.gw / 2, conf.gw / 2))
    axes.set_ylim((0, conf.gh))


def bool_to_bgr(mat):
    return cv2.cvtColor(mat.astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)


f, ax = plt.subplots(3, 2)

ransac_output = ransac_output.astype(np.uint8) * 255
occ = cv2.cvtColor(occ, cv2.COLOR_GRAY2BGR)

ax[0][0].set_title("original image")
ax[0][0].imshow(image[:, :, [2, 1, 0]])  # [100:, :, [2, 1, 0]])

ax[0][1].set_title("segmented (ransac + hsv)")
ax[0][1].imshow(cv2.cvtColor(ransac_output, cv2.COLOR_GRAY2RGB))

# show_pc(ax[1][0], drive_rpc, drive_conf, "driveable cloud")
# show_pc(ax[1][1], block_rpc, drive_conf, "obstacle cloud")

ax[2][0].set_title("bilinear interp (2 ms)")
ax[2][0].imshow(occ)
ax[2][1].set_title("line of sight (25 ms)")
# ax[2][1].imshow(cv2.cvtColor(los_grid, cv2.COLOR_GRAY2BGR))

plt.show()

# c1, c2, c3 = best_coeffs
# ys, xs = np.indices((h, w))
# z_pred = c1 * xs + c2 * ys + c3

# fig = plt.figure(figsize=(10, 7))
# ax = fig.add_subplot(111, projection="3d")

# mask = input_data != -np.inf
# ax.scatter(xs[mask], ys[mask], input_data[mask], c='b', s=2, label='Data')
# ax.plot_surface(xs, ys, z_pred, color='r', alpha=0.4)

# ax.set_title("RANSAC Plane Fit")
# ax.set_xlabel("X")
# ax.set_ylabel("Y")
# ax.set_zlabel("Depth")
# plt.legend()
# plt.show()

# TODO? double pass ransac, pick points within ok zone of first half of iterations for the second half of iterations
