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
    print(linearize.endpoints(mask))
# End test simple