import numpy as np

"""
The real occupancy grid is 100x100 pixels (50 mm x 50 mm squares)
"""
grid_resolution = 50  # 50 mm
ground_occ_shape = (100, 100)


def x_img2x_gnd(x_img, depth_reading, cx: int = 320, fx: int = 640):
    """
    Convert pixel x-coordinate to ground X (mm).
    depth_reading must be in mm.
    """
    return (x_img - cx) * depth_reading / fx


def real_world2data_structure(potential_x, potential_z, occ_grid_shape=ground_occ_shape):
    """
    potential_x and potential_z are in mm.
    Convert ground-plane coordinates to occupancy grid indices.
    """
    occ_rows, occ_cols = occ_grid_shape

    grid_row = potential_z // grid_resolution
    grid_col = potential_x // grid_resolution + occ_cols // 2

    return int(grid_row), int(grid_col)


def image_to_ground(image_drivable_grid, depth_map, fx=640, fy=640):
    """
    Transform pixel-wise drivable image into a ground-plane occupancy grid.
    image_drivable_grid: binary mask, 1 = drivable, 0 = obstacle
    depth_map: same size as grid, depth in mm
    """
    height, width = depth_map.shape
    cx = width // 2

    # Default: all "1" means obstacle by your convention (blocked)
    ground_occ_grid = np.ones(ground_occ_shape, dtype=np.uint8)

    for r in range(height):
        for c in range(width):

            if image_drivable_grid[r][c] == 0:
                continue

            # Convert pixel to horizontal ground X
            x_img = c     # FIXED: should use column, not row
            depth_reading = depth_map[r][c]

            # Ignore invalid depth
            if depth_reading <= 0:
                continue

            potential_x = x_img2x_gnd(x_img, depth_reading, cx=cx, fx=fx)
            potential_z = depth_reading  # your Z is forward distance (mm)

            grid_row, grid_col = real_world2data_structure(
                potential_x, potential_z, ground_occ_grid.shape
            )

            # Bounds check FIX
            if (
                grid_row < 0 or grid_row >= ground_occ_grid.shape[0] or
                grid_col < 0 or grid_col >= ground_occ_grid.shape[1]
            ):
                continue

            # Mark as drivable (0)
            ground_occ_grid[grid_row, grid_col] = 0

            # Fill leftwards if consecutive vertical pixels are drivable
            if c > 0:
                if image_drivable_grid[r, c] == 1 and image_drivable_grid[r, c-1] == 1:
                    temp_c = grid_col
                    # Spread until you hit a cell already cleared or out of bounds
                    while temp_c >= 0 and ground_occ_grid[grid_row, temp_c] == 1:
                        ground_occ_grid[grid_row, temp_c] = 0
                        temp_c -= 1

    return ground_occ_grid
