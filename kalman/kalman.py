"""
This file holds all functionality for dealing with the Kalman Filter for Lane Detection. The variables being
predicted are
- Left lane intercept (with bottom of image)
- Right lane intercept
- Left lane angle (0 degrees is straight, positive values are angling right, and negative values are angling left)
- Right lane angle 

Internally, we also hold velocities and accelerations for
each of these values to aid in prediction.

"""

import numpy as np
from scipy.linalg import block_diag

"""
There is no control variable in this case

x_hat (new) = F @ x_hat

"""


# State variables
l_intercept = 0
l_intercept_dot = 0
l_intercept_dot_dot = 0
l_angle = 0
l_angle_dot = 0
l_angle_dot_dot = 0
r_intercept = 0
r_intercept_dot = 0
r_intercept_dot_dot = 0
r_angle = 0
r_angle_dot = 0
r_angle_dot_dot = 0

cov = np.eye(12)

# Noise Covariance (will have to tune these)
Q = np.eye(12)*0.001 # Additive process noise covariance
R = np.eye(4)*0.0001 # Additive measurement noise covariance
I = np.eye(12)

# Needed for prediction model
def calc_F(dt):
    model = np.array([
        [1, dt, 0.5*dt*dt],
        [0, 1, dt],
        [0, 0, 1]
    ])
    F = block_diag(model, model, model, model)
    return F

# Measurement is only of the intercepts and angles
def calc_H():
    H = np.zeros((4,12))
    H[0,0] = 1
    H[1,3] = 1
    H[2,6] = 1
    H[3,9] = 1
    return H

def kalman_init(measurement, R_est = R):
    global l_intercept, l_angle, r_intercept, r_angle, R
    l_intercept, l_angle, r_intercept, r_angle = measurement
    R = R_est

def kalman_estimate(mean, covariance, measurement, dt):
    F = calc_F(dt)
    H = calc_H()
    pred_mean = F @ mean # (12,)
    pred_cov = F @ covariance @ F.T + Q # (12,12)
    innovation = measurement - H @ pred_mean # (4,)
    innovation_cov = H @ pred_cov @ H.T + R # (4,4)
    filter_gain = pred_cov @ H.T @ np.linalg.inv(innovation_cov) # (12,4)
    #print("Filter Gain = ", filter_gain)
    corr_mean = pred_mean + filter_gain @ innovation # (12,)
    corr_cov = (I - filter_gain @ H) @ pred_cov # (12,12)
    return corr_mean, corr_cov

# measurement is a 4 element array with [left_intercept, l_angle, r_intercept, r_angle]
# dt is the length of the time step between measurements
def kalman(measurement, dt):
    global l_intercept, l_intercept_dot, l_intercept_dot_dot, l_angle, l_angle_dot, l_angle_dot_dot, r_intercept, r_intercept_dot, r_intercept_dot_dot, r_angle, r_angle_dot, r_angle_dot_dot, cov
    mean = np.array([
        l_intercept,
        l_intercept_dot,
        l_intercept_dot_dot,
        l_angle,
        l_angle_dot,
        l_angle_dot_dot,
        r_intercept,
        r_intercept_dot,
        r_intercept_dot_dot,
        r_angle,
        r_angle_dot,
        r_angle_dot_dot
    ])
    mean, cov = kalman_estimate(mean, cov, measurement, dt)
    l_intercept, l_intercept_dot, l_intercept_dot_dot, l_angle, l_angle_dot, l_angle_dot_dot, r_intercept, r_intercept_dot, r_intercept_dot_dot, r_angle, r_angle_dot, r_angle_dot_dot = mean
    return l_intercept, l_angle, r_intercept, r_angle
