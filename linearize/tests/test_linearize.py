import unittest
from linearize import linearize
import numpy as np

def test_simple():
    # A simple mask
    mask = np.array([
        [0, 1, 0, 1],
        [0, 1, 0, 1],
        [0, 1, 0, 1],
        [0, 1, 0, 1],
        [0, 1, 0, 1],
        [0, 1, 0, 1]
    ])
    # First, make a copy of the mask to avoid the change of the original structure
    mask_matrix_copy = mask
    # Get the endpoints of the mask
    endpoints_info = linearize.endpoints(mask_matrix_copy)
    # Calculate the angle position
    angle_positions = linearize.compute_angle(mask_matrix_copy, endpoints_info)
# End test simple