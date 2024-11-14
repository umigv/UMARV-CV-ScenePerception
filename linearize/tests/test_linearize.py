import unittest
from linearize import linearize
import numpy as np
import math


def test_simple_no_angle():
    """A simple mask"""
    mask = np.array(
        [
            [0, 1, 0, 1],
            [0, 1, 0, 1],
            [0, 1, 0, 1],
            [0, 1, 0, 1],
            [0, 1, 0, 1],
            [0, 1, 0, 1],
        ]
    )
    # First, make a copy of the mask to avoid the change of the original structure
    mask_matrix_copy = mask
    # Get the endpoints of the mask
    endpoints_info = linearize.endpoints(mask_matrix_copy)
    # Calculate the angle position
    angle_positions = linearize.compute_angle(mask_matrix_copy, endpoints_info)
    # Check if the angles equal
    assert np.array_equal(angle_positions, np.array([0.0, 0.0], dtype=np.float64))


# End test simple


def test_simple_angle():
    """A simple test with the angle degree."""
    mask = np.array(
        [
            [1, 1, 0, 1],
            [0, 1, 0, 1],
            [0, 1, 0, 1],
            [0, 1, 0, 1],
            [0, 1, 0, 1],
            [0, 1, 0, 1],
        ]
    )
    # First, make a copy of the mask to avoid the change of the original structure
    mask_matrix_copy = mask
    # Get the endpoints of the mask
    endpoints_info = linearize.endpoints(mask_matrix_copy)
    # Calculate the angle position
    angle_positions = linearize.compute_angle(mask_matrix_copy, endpoints_info)
    assert math.isclose(angle_positions[0], -9.462, abs_tol=10**-3)
    assert angle_positions[1] == 0


# End test simple


def test_large_mask():
    """A simple test with the angle degree."""
    mask = np.array(
        [
            [1, 1, 0, 0, 0, 0, 0, 1, 1],
            [1, 1, 0, 0, 0, 0, 0, 1, 1],
            [1, 1, 0, 0, 0, 0, 0, 1, 1],
            [1, 1, 0, 0, 0, 0, 0, 1, 1],
            [1, 1, 0, 0, 0, 0, 0, 1, 1],
            [1, 1, 0, 0, 0, 0, 0, 1, 1],
            [1, 1, 0, 0, 0, 0, 0, 1, 1],
            [1, 1, 0, 0, 0, 0, 0, 1, 1],
            [1, 1, 0, 0, 0, 0, 0, 1, 1],
            [1, 1, 0, 0, 0, 0, 0, 1, 1],
            [1, 1, 0, 0, 0, 0, 0, 1, 1],
            [1, 1, 0, 0, 0, 0, 0, 1, 1],
            [1, 1, 0, 0, 0, 0, 0, 1, 1],
            [1, 1, 0, 0, 0, 0, 0, 1, 1],
            [1, 1, 0, 0, 0, 0, 0, 1, 1],
            [1, 1, 0, 0, 0, 0, 0, 1, 1],
            [1, 1, 0, 0, 0, 0, 0, 1, 1],
            [1, 1, 0, 0, 0, 0, 0, 1, 1],
            [1, 1, 0, 0, 0, 0, 0, 1, 1],
            [1, 1, 0, 0, 0, 0, 0, 1, 1],
        ]
    )
    # First, make a copy of the mask to avoid the change of the original structure
    mask_matrix_copy = mask
    # Get the endpoints of the mask
    endpoints_info = linearize.endpoints(mask_matrix_copy)
    # Calculate the angle position
    angle_positions = linearize.compute_angle(mask_matrix_copy, endpoints_info)
    # Check if the angle degree is close enough
    assert math.isclose(angle_positions[0], -2.862, abs_tol=10**-3)
    assert math.isclose(angle_positions[1], 2.862, abs_tol=10**-3)


# End test simple
