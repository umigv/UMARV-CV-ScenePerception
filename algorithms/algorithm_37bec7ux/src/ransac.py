import numpy as np  # matrices
import matplotlib.pyplot as plt
import h5py  # reading data
import skimage
import random
import time
import math

# PARAMETERS

filename = "res/19_11_10.hdf5"
frame_number = 400

iters = 100
kernel = (3, 3) # kernel is rows, columns
# with normalise tolerance 0.3 works well
# tolerance = 300
tolerance = 0.23

# INPUT FILTERING (@the2nake)

# take in the 720p? data (some high-res data)
f = h5py.File(filename, "r")
# print(list(f.keys()))
depth_map = f["depth_maps"][frame_number]
image = f["images"][frame_number]
image = image[:, 0:int(image.shape[1]/2)]

f.close()

# START

start = time.perf_counter()

depth_map = np.where(
    np.isinf(depth_map) | np.isnan(depth_map), -np.inf, depth_map
)  # swap inf, nan to -inf

# normalise (?)
max_depth = int(depth_map.max())
depth_map = max_depth / depth_map


# decide on a pooling method (avg, max)
# pool to reduce dimension of data (numpy)
def average(depth_map, kernel: tuple[int, int]):
    h, w = depth_map.shape
    w -= w % kernel[1]
    h -= h % kernel[0]
    depth_map = depth_map[0:h, 0:w]
    return skimage.measure.block_reduce(depth_map, kernel, np.mean)


input_data = average(
    depth_map, kernel
)  # []  # 2d matrix, indices corresponding to pixels, data is depth

# RANSAC (@kjosh491)
# make the algorithm work for any input dimensions
# output the best plane's coefficients with respect to [px, py, depth] coordinate system

h, w = input_data.shape


def sample_points():
    # check A matrix will turn out with rank < 3 (check x and y values)
    b = np.array([[0.0], [0.0], [0.0]])
    A = np.ndarray((3, 3))
    # repeat sampling instead of selecting using argwhere (don't process the whole array)
    while True:
        for i in range(3):
            row = -1
            col = -1
            while row < 0 or col < 0 or np.isneginf(input_data[row][col]):
                row = math.floor(random.random() * h)
                col = math.floor(random.random() * w)
            b[i] = [input_data[row][col]]
            A[i] = [float(col), float(row), 1.0]
        if np.linalg.matrix_rank(A) == 3:
            break
    return A, b


def calculate_plane(A, b):
    """Solve for plane coefficients [c1, c2, c3] in z = c1*x + c2*y + c3"""
    coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return coeffs  # [c1, c2, c3]


# tolerance is percentage of maximum depth
def calculate_metric(c1, c2, c3, tol=tolerance):
    h, w = input_data.shape
    ys, xs = np.indices((h, w))
    z_pred = c1 * xs + c2 * ys + c3
    mask = input_data != -np.inf
    errors = np.abs(z_pred[mask] - input_data[mask])
    return np.sum(errors < tol)


best = 0
best_coeffs = [0, 0, 0]

for i in range(iters):
    A, b = sample_points()
    c1, c2, c3 = calculate_plane(A, b)
    metric = calculate_metric(c1, c2, c3)
    if metric > best:
        best = metric
        best_coeffs = [c1, c2, c3]
        
print("Best plane coefficients:", best_coeffs)

# now we have the best plane fit

# OUTPUT FORMAT (@the2nake)

def get_mask(c1, c2, c3, tol=tolerance):
    h, w = depth_map.shape
    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    Z = abs(c1 * X + c2 * Y + c3 - depth_map)
    return Z < tol


best_mask = get_mask(best_coeffs[0] / kernel[1], best_coeffs[1] / kernel[0], best_coeffs[2])
# output_mask = skimage.transform.rescale(best_mask, kernel)

end = time.perf_counter()
f, axarr = plt.subplots(2,1)
axarr[0].imshow(image)
axarr[1].imshow(best_mask, interpolation='nearest')
plt.show()
print(f"{1000 * (end - start)} ms per frame")

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
