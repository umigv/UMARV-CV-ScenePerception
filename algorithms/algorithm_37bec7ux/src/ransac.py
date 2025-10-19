import numpy  # matrices
import matplotlib
import h5py  # reading data


# INPUT FILTERING (@the2nake)
take in the 720p? data (some high-res data)
decide on a pooling method (avg, max)
pool to reduce dimension of data (numpy)

input_data = [] # 2d matrix, indices corresponding to pixels, data is depth

# RANSAC (@kjosh491)
# make the algorithm work for any input dimensions
# output the best plane's coefficients with respect to [px, py, depth] coordinate system

iters = 100

def calculate_plane():
    # c1 * x + c2 * y + c3 = z
    pass


def sample_points():
    # check A matrix will turn out with rank < 3 (check x and y values)
    pass

def calculate_metric():
    # metric = abs(c1 * x + c2 * y - z - c3)
    pass

best = 0
best_coeffs = [0, 0, 0]

for i in range(iters):
    [A, b] = sample_points()
    [c1, c2, c3] = calculate_plane(A, b)
    metric = calculate_metric(c1, c2, c3)
    if (metric > best):
        best = metric
        best_coeffs = [c1, c2, c3]

# now we have the best plane fit

# TODO: MATPLOTLIB VISUALISATION TO CHECK PLANE VALIDITY ON ?RANDOM? POINT DATA

# OUTPUT FORMAT (@the2nake)

# figure out corresponding to pixels
# essentially one last iteration of ransac without the sampling

for each pixel:
    check if it is an outlier
    create a 2d matrix of boolean values with the same dimensions as the pooled data

# UPSIZE THIS TO THE ORIGINAL HIGH_RES FORMAT (NEAREST NEIGHBOUR)