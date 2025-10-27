import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import math
import time

# PARAMETERS
iters = 100

#Test Data

h, w = 80, 80
xs, ys = np.meshgrid(np.arange(w), np.arange(h))
true_plane = 0.3 * xs + 0.5 * ys + 0.2
noise = np.random.normal(0, 1, size=true_plane.shape)
depth_map = true_plane + noise

# introduce some outliers
mask_outliers = np.random.rand(h, w) < 0.05
depth_map[mask_outliers] = -np.inf  # simulate invalid data


max_depth = depth_map[np.isfinite(depth_map)].max()
depth_map /= max_depth

# Use same variable name as original
input_data = depth_map

#RANSAC implementation

def sample_points_1():
    # check A matrix will turn out with rank < 3 (check x and y values)
    b = np.array([[0.0], [0.0], [0.0]])
    A = np.ndarray((3, 3))
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
    print(A, b)
    return A, b

def sample_points_2():
    """Randomly sample 3 unique points (x, y, z) from the input depth map"""
    h, w = input_data.shape
    valid = np.argwhere(input_data != -np.inf)
    idx = np.random.choice(len(valid), 3, replace=False)
    pts = valid[idx]
    A = np.column_stack((pts[:, 1], pts[:, 0], np.ones(3)))  # [x y 1]
    b = input_data[pts[:, 0], pts[:, 1]]
    print(A, b)
    return A, b

def sample_points():
    return sample_points_2()


def calculate_plane(A, b):
    # c1 * x + c2 * y + c3 = z
    return np.dot(np.linalg.inv(A), b)


# tolerance is percentage of maximum depth
def calculate_metric(c1, c2, c3, tolerance=0.1):
    h, w = input_data.shape
    ys, xs = np.indices((h, w))
    z_pred = c1 * xs + c2 * ys + c3
    mask = input_data != -np.inf
    errors = np.abs(z_pred[mask] - input_data[mask])
    inliers = np.sum(errors < tolerance)
    return inliers

best = 0
best_coeffs = [0, 0, 0]

start = time.perf_counter()

for i in range(iters):
    A, b = sample_points()
    c1, c2, c3 = calculate_plane(A, b)
    metric = calculate_metric(c1, c2, c3)
    if metric > best:
        best = metric
        best_coeffs = [c1, c2, c3]


end = time.perf_counter()

print(1000 * (end - start))

#MatPlot Visualization
c1, c2, c3 = best_coeffs
ys, xs = np.indices((h, w))
z_pred = c1 * xs + c2 * ys + c3

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")

mask = input_data != -np.inf
ax.scatter(xs[mask], ys[mask], input_data[mask], c='b', s=2, label='Data')
ax.plot_surface(xs, ys, z_pred, color='r', alpha=0.4)

ax.set_title("RANSAC Plane Fi ")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Depth")
plt.legend()
plt.show()