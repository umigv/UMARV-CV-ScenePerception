# RANSAC

### by kjosh491, the2nake

## Algorithm description

Creates an occupancy grid of obstacles using RGB-D data, filtering for lane lines on the ground plane.

## Notes

Typical pipeline setup:

`ransac.plane.clean_depths` --> `ransac.plane.hsv_and_ransac` --> `ransac.plane.real_coeffs` -->

`ransac.occu.create_point_cloud` --> `ransac.occu.pixel_to-real` --> `ransac.occu.occupancy_grid`

This approach is extremely inefficient and should later be replaced by a homography matrix with points of interest based on the occupancy grid specification. Find the source points for the destination `(-h/c, 2h/c)` `(h/c, 2h/c)` `(-h/c, 0)` `(h/c, 0)` then perform either `cv.warpPerspective` or bilinear interpolation with the relevant formulas derived by hand (which would be more useful for performing the inverse operation later requested).

## Dependencies

- check `docs/requirements.txt`
- make sure to download test files (check `res/README.md`)
