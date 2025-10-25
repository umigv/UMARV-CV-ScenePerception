import numpy as np  # matrices
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import h5py  # reading data

# PARAMETERS

filename = "res/19_11_10.hdf5"
frame_number = 0

iters = 100

# INPUT FILTERING (@the2nake)

# take in the 720p? data (some high-res data)
f = h5py.File(filename, "r")
# keys = list(f.keys())
depth_map = f["depth_maps"][frame_number]
depth_map = np.where(np.isinf(depth_map) | np.isnan(depth_map), -np.inf, depth_map)  # swap inf, nan to -inf

# normalise (?)
max_depth = int(depth_map.max())
depth_map /= max_depth

# decide on a pooling method (avg, max)
# pool to reduce dimension of data (numpy)

input_data = depth_map # []  # 2d matrix, indices corresponding to pixels, data is depth

# RANSAC (@kjosh491)
# make the algorithm work for any input dimensions
# output the best plane's coefficients with respect to [px, py, depth] coordinate system


def calculate_plane(A,b):
    # c1 * x + c2 * y + c3 = z
     coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return coeffs  # [c1, c2, c3]
    


def sample_points():
    # check A matrix will turn out with rank < 3 (check x and y values)
    h, w = input_data.shape
    valid = np.argwhere(input_data != -np.inf)
    idx = np.random.choice(len(valid), 3, replace=False)
    pts = valid[idx]
    A = np.column_stack((pts[:, 1], pts[:, 0], np.ones(3)))  # [x y 1]
    b = input_data[pts[:, 0], pts[:, 1]]
    return A, b
    


def calculate_metric(c1,c2,c3):
    # metric = abs(c1 * x + c2 * y - z - c3)
    h, w = input_data.shape
    ys, xs = np.indices((h, w))
    z_pred = c1 * xs + c2 * ys + c3
    mask = input_data != -np.inf
    errors = np.abs(z_pred[mask] - input_data[mask])
    inliers = np.sum(errors < 0.02)  # tolerance
    return inliers
   


best = 0
best_coeffs = [0, 0, 0]

for i in range(iters):
    [A, b] = sample_points()
    [c1, c2, c3] = calculate_plane(A, b)
    metric = calculate_metric(c1, c2, c3)
    if metric > best:
        best = metric
        best_coeffs = [c1, c2, c3]
        
print("Best plane coefficients:", best_coeffs)

# now we have the best plane fit

# TODO: MATPLOTLIB VISUALISATION TO CHECK PLANE VALIDITY ON ?RANDOM? POINT DATA
c1, c2, c3 = best_coeffs
ys, xs = np.indices((h, w))
z_pred = c1 * xs + c2 * ys + c3

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")

mask = input_data != -np.inf
ax.scatter(xs[mask], ys[mask], input_data[mask], c='b', s=2, label='Data')
ax.plot_surface(xs, ys, z_pred, color='r', alpha=0.4)

ax.set_title("RANSAC Plane Fit")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Depth")
plt.legend()
plt.show()
# OUTPUT FORMAT (@the2nake)

# figure out corresponding to pixels
# essentially one last iteration of ransac without the sampling

# for each pixel:
#     check if it is an outlier
#     create a 2d matrix of boolean values with the same dimensions as the pooled data

# UPSIZE THIS TO THE ORIGINAL HIGH_RES FORMAT (NEAREST NEIGHBOUR)
