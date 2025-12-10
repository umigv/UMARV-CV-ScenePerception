import numpy as np  # matrices
import matplotlib.pyplot as plt
import h5py  # reading data
import random
import time
import math
import cv2

import ransac.plane, ransac.occu

# PARAMETERS

filename = "res/perspective_test.svo2.hdf5"
frame_number = -1

iters = 50
kernel = (1, 16)  # kernel is rows, columns
tolerance = 0.1

# INPUT FILTERING (@the2nake)

f = h5py.File(filename, "r")
# print(list(f.keys()))
frames = len(f["depth_maps"])
if frame_number < 0:
    frame_number = int(math.floor(random.random() * frames))
    print(f"Using randomised frame number: {frame_number}")
elif frame_number >= frames:
    frame_number = frames - 1


raw_depths = f["depth_maps"][frame_number]
depth_map = f["depth_maps"][frame_number]
image = f["images"][frame_number]
image = image[:, 0 : int(image.shape[1] / 2)]

f.close()

# START

start = time.perf_counter()

cleaned_depths = ransac.plane.clean_depths(raw_depths)
driveable, ransac_coeffs = ransac.plane.hsv_and_ransac(
    image, cleaned_depths, 60, (1, 16), 0.15
)
ransac_output = driveable  # [100:, :]

fx = 360

h, w = depth_map.shape
intrinsics = ransac.CameraIntrinsics(w / 2, h / 2, fx, fx)
real = ransac.plane.real_coeffs(ransac_coeffs, intrinsics)
angle = ransac.plane.real_angle(real)

driveable_ppc = ransac.occu.create_point_cloud(driveable, cleaned_depths)
driveable_rpc = ransac.occu.pixel_to_real(driveable_ppc, real, intrinsics)

obstacle_ppc = ransac.occu.create_point_cloud(driveable != 1, cleaned_depths)
obstacle_rpc = ransac.occu.pixel_to_real(obstacle_ppc, real, intrinsics)

driveable_conf = ransac.OccupancyGridConfiguration(5000, 5000, 50, thres=5)  # in millimetres
obstacle_conf = ransac.OccupancyGridConfiguration(5000, 5000, 50, thres=1)  # in millimetres
driveable_occ = ransac.occu.occupancy_grid(driveable_rpc, driveable_conf)
obstacle_occ = ransac.occu.occupancy_grid(obstacle_rpc, obstacle_conf)

merged = ransac.occu.merge(driveable_occ, obstacle_occ)

occ_h, occ_w = merged.shape
cam = ransac.VirtualCamera(occ_h - 1, occ_w // 2, math.pi / 2, math.pi / 2)
los_grid = ransac.occu.create_los_grid(merged, [cam]) # remove cam to use morphology technique (faster)
end = time.perf_counter()

# DISPLAY DATA

print("coeffs: ", ransac_coeffs)
print("angle: ", math.degrees(angle))
print(driveable_rpc)

print("-----")

# PLOT THINGS

def show_pc(axes, cloud, conf: ransac.OccupancyGridConfiguration, name: str = "point cloud"):
    axes.set_title(name)
    axes.scatter(cloud[:, 0], cloud[:, 2], s=0.01)
    axes.set_aspect("equal", adjustable="box")
    axes.set_xlim((-conf.gw / 2, conf.gw / 2))
    axes.set_ylim((0, conf.gh))

def bool_to_bgr(mat):
    return cv2.cvtColor(mat.astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)

f, ax = plt.subplots(3, 2)

ransac_output = ransac_output.astype(np.uint8) * 255
merged = cv2.cvtColor(merged, cv2.COLOR_GRAY2BGR)

ax[0][0].set_title("original image")
ax[0][0].imshow(image[:, :, [2, 1, 0]])  # [100:, :, [2, 1, 0]])

ax[0][1].set_title("segmented (ransac + hsv)")
ax[0][1].imshow(cv2.cvtColor(ransac_output, cv2.COLOR_GRAY2RGB))

show_pc(ax[1][0], driveable_rpc, driveable_conf, "driveable cloud")
show_pc(ax[1][1], obstacle_rpc, driveable_conf, "obstacle cloud")

ax[2][0].set_title("merged area")
ax[2][0].imshow(merged)
ax[2][1].set_title("line of sight")
ax[2][1].imshow(cv2.cvtColor(los_grid, cv2.COLOR_GRAY2BGR))

plt.show()

print(f"-----\n{1000 * (end - start)} ms per frame")

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
