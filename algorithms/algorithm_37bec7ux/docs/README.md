# RANSAC
### by kjosh491, the2nake

# Model description
{model description here}

# Notes
* make sure to download the `res/19_11_10.hdf5` test file

* input: int[w x h] extracted from .svo2 or live from ZED camera
* params:
  * pooling (int): size of pooling square pooling kernel
  * method (bool): use max or average pool method
* output: bool[n x m] occupancy grid with dimensions as needed
