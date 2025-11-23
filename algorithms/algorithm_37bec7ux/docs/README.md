# RANSAC

### by kjosh491, the2nake

## Algorithm description

Creates an occupancy grid of obstacles using RGB-D data, filtering for lane lines on the ground plane.

## Notes

Typical pipeline setup:

`ransac.plane.clean_depths` --> `ransac.plane.hsv_and_ransac` --> `ransac.plane.real_coeffs` -->

`ransac.occu.create_point_cloud` --> `ransac.occu.pixel_to-real` --> `ransac.occu.occupancy_grid`

## Dependencies

- check `docs/requirements.txt`
- make sure to download the `res/19_11_10.hdf5` test file (check `res/README.md`)
