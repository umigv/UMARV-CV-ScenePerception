import numpy as np  # matrices
import matplotlib.pyplot as plt
import h5py  # reading data
import skimage  # image filter
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
# with normalise tolerance 0.3 works well
# tolerance = 300
tolerance = 0.1

# INPUT FILTERING (@the2nake)

# take in the 720p? data (some high-res data)
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

h, w = depth_map.shape

f.close()

# START

start = time.perf_counter()

cleaned_depths = ransac.plane.clean_depths(raw_depths)
ransac_raw, ransac_coeffs = ransac.plane.hsv_and_ransac(
    image, cleaned_depths, 60, (1, 16), 0.1
)
ransac_output = ransac_raw  # [100:, :]

fx = 300
intrinsics = ransac.CameraIntrinsics(w / 2, h / 2, fx, fx)
real = ransac.plane.real_coeffs(ransac_coeffs, intrinsics)
angle = ransac.plane.real_angle(real)

pixel_pc = ransac.occu.create_point_cloud(ransac_raw, cleaned_depths)  # slow
real_pc = ransac.occu.pixel_to_real(pixel_pc, real, intrinsics)  # slow

end = time.perf_counter()


print("coeffs: ", ransac_coeffs)
print("angle: ", math.degrees(angle))
# print(real_pc)

print("-----")

f, axarr = plt.subplots(3, 1)

ransac_output = ransac_output.astype(np.uint8) * 255

axarr[0].imshow(image[:, :, [2, 1, 0]])  # [100:, :, [2, 1, 0]])
axarr[1].imshow(cv2.cvtColor(ransac_output, cv2.COLOR_GRAY2RGB))
axarr[2].scatter(real_pc[:, 0], real_pc[:, 2], s=0.01)

xlim = axarr[2].get_xlim()
xlim_max = max(abs(xlim[0]), abs(xlim[1]))
axarr[2].set_xlim((-xlim_max, xlim_max))

axarr[2].set_xlim((-1000, 1000))
axarr[2].set_ylim((0, 2000))
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
