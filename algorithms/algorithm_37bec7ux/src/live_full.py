########################################################################
# Live RANSAC → Occupancy Grid Visualization from ZED (mm depth units)
########################################################################

import sys
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
from signal import signal, SIGINT
import pyzed.sl as sl

import ransac
import transform

########################################################################
# CTRL+C handler
########################################################################

cam = sl.Camera()

def handler(sig, frame):
    print("Shutting down...")
    cam.close()
    sys.exit(0)

signal(SIGINT, handler)

########################################################################
# MAIN
########################################################################

def main():
    #########################################################
    # ZED INIT
    #########################################################
    init = sl.InitParameters()
    init.depth_mode = sl.DEPTH_MODE.NEURAL
    init.async_image_retrieval = False
    init.coordinate_units = sl.UNIT.MILLIMETER   # << Depth in millimeters

    status = cam.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        print("Camera Open:", status)
        return

    runtime = sl.RuntimeParameters()

    # Retrieve intrinsics
    cam_info = cam.get_camera_information()
    calib = cam_info.camera_configuration.calibration_parameters

    fx = calib.left_cam.fx
    fy = calib.left_cam.fy
    cx = calib.left_cam.cx
    cy = calib.left_cam.cy

    print("\nZED Intrinsics (LEFT camera):")
    print(f"fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}\n")

    #########################################################
    # Buffers
    #########################################################
    image = sl.Mat()
    depth = sl.Mat()

    # Matplotlib figure for occupancy grid
    plt.ion()
    fig, ax = plt.subplots(figsize=(6,4))
    im = None

    scale = 0.05  # 0.05 meters per cell

    key = 0
    while key != ord('q'):

        #########################################################
        # Grab frame
        #########################################################
        err = cam.grab(runtime)
        if err != sl.ERROR_CODE.SUCCESS:
            print("Grab:", err)
            continue

        cam.retrieve_image(image, sl.VIEW.LEFT, sl.MEM.CPU)
        cam.retrieve_measure(depth, sl.MEASURE.DEPTH, sl.MEM.CPU)

        img = image.get_data()[:, :, :3]
        depth_arr = depth.get_data()

        #########################################################
        # Clean depth, run RANSAC
        #########################################################
        depth_clean = ransac.clean_depths(depth_arr)

        ransac_output, _ = ransac.ransac(
            depth_clean,
            max_iters=60,
            kernel=(1, 16),
            tolerance=0.1
        )

        #########################################################
        # Occupancy grid transform
        #########################################################
        occupancy_grid = transform.image_to_ground(
            ransac_output,
            depth_clean,
            fx, fy
        )

        h, w = occupancy_grid.shape

        #########################################################
        # Display occupancy grid
        #########################################################
        if im is None:
            im = ax.imshow(
                occupancy_grid,
                cmap="gray_r",             # 0=white, 1=black
                interpolation="nearest",
                extent=[0, w * scale, 0, h * scale],
                origin="lower"
            )
            ax.set_xlabel("meters")
            ax.set_ylabel("meters")
        else:
            im.set_data(occupancy_grid)
            im.set_extent([0, w * scale, 0, h * scale])

        plt.pause(0.001)

        #########################################################
        # Show camera feed
        #########################################################
        cv2.imshow("View (Left Camera)", img)
        key = cv2.waitKey(1)

    #########################################################
    # Cleanup
    #########################################################
    plt.ioff()
    cv2.destroyAllWindows()
    cam.close()


if __name__ == "__main__":
    main()
