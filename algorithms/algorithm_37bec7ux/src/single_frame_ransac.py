import numpy as np  # matrices
import matplotlib.pyplot as plt
import h5py  # reading data
import skimage  # image filter
import random
import time
import math
import ransac

# PARAMETERS

filename = "res/19_11_10.hdf5"
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
if frame_number < 0:
    frame_number = int(math.floor(random.random() * len(f["depth_maps"])))
    print(f"Using randomised frame number: {frame_number}")


raw_depths = f["depth_maps"][frame_number]
depth_map = f["depth_maps"][frame_number]
image = f["images"][frame_number]
image = image[:, 0 : int(image.shape[1] / 2)]

h, w = depth_map.shape

f.close()

# START

start = time.perf_counter()
cleaned_depths = ransac.clean_depths(raw_depths)
ransac_output, ransac_coeffs = ransac.ransac(cleaned_depths, 60, (1, 16), 0.1)
ransac_output = ransac_output[100:, :]
real = ransac.real_coeffs(ransac_coeffs, w / 2, h / 2, 1057 / 2, 1057 / 2)
angle = ransac.real_angle(real)
end = time.perf_counter()

f, axarr = plt.subplots(2, 1)
axarr[0].imshow(image[100:, :, [2, 1, 0]])
axarr[1].imshow(ransac_output, interpolation="nearest")
plt.show()
print(f"{1000 * (end - start)} ms per frame")

print(np.asarray(real))
print(angle)

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
