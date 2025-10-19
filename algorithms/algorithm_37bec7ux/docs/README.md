# RANSAC
### by kjosh491, the2nake

# Model description
{model description here}

# Notes

## I/O

* input: int[w x h] extracted from .svo2 or live from ZED camera
* params:
  * pooling (int): size of pooling square pooling kernel
  * method (bool): use max or average pool method
* output: bool[n x m] occupancy grid with dimensions as needed

## Dependencies

* python
  * matplotlib >= 3.5.0
  * h5py >= 3.0.0
* make sure to download the `res/19_11_10.hdf5` test file (check `src/README.md`)
