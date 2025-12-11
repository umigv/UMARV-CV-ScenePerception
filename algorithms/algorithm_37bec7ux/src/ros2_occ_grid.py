########################################################################
#
# Copyright (c) 2022, STEREOLABS.
#
# All rights reserved.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
########################################################################

import sys
import pyzed.sl as sl
from signal import signal, SIGINT
import argparse
import os
import cv2
import ransac.plane, ransac.occu
import numpy as np
import math
# >>> ros2 change
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, MapMetaData
# <<< ros2 end of change


cam = sl.Camera()

# >>> ros2 change
class OccGridPublisher(Node):
    def __init__(self, width: int, height: int, resolution: float):
        super().__init__('occ_grid_publisher')
        self.pub = self.create_publisher(OccupancyGrid, 'occ_grid', 10)
        self.width = width
        self.height = height
        self.resolution = resolution

    def publish(self, grid_np):
        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'odom'

        info = MapMetaData()
        info.width = self.width
        info.height = self.height
        info.resolution = self.resolution
        msg.info = info

        # Convert internal 0/127/255 encoding to ROS -1/0/100
        flat = grid_np.astype('uint8')
        ros = np.full(flat.shape, -1, dtype=np.int8)
        ros[flat == 0] = 100   # occupied
        ros[flat == 255] = 0   # free
        msg.data = ros.flatten().tolist()

        self.pub.publish(msg)
# <<< ros2 end of change


# Handler to deal with CTRL+C properly
def handler(signal_received, frame):
    cam.disable_recording()
    cam.close()
    sys.exit(0)


signal(SIGINT, handler)


def print_params(calibration_params: sl.CalibrationParameters):
    # LEFT CAMERA intrinsics
    fx_left = calibration_params.left_cam.fx
    fy_left = calibration_params.left_cam.fy
    cx_left = calibration_params.left_cam.cx
    cy_left = calibration_params.left_cam.cy

    # RIGHT CAMERA intrinsics
    fx_right = calibration_params.right_cam.fx
    fy_right = calibration_params.right_cam.fy
    cx_right = calibration_params.right_cam.cx
    cy_right = calibration_params.right_cam.cy

    # Translation (baseline) between left and right camera
    tx = calibration_params.stereo_transform.get_translation().get()[0]

    # Print results
    print("\n--- ZED Camera Calibration Parameters ---")
    print("Left Camera Intrinsics:")
    print(f"  fx = {fx_left:.3f}")
    print(f"  fy = {fy_left:.3f}")
    print(f"  cx = {cx_left:.3f}")
    print(f"  cy = {cy_left:.3f}\n")

    print("Right Camera Intrinsics:")
    print(f"  fx = {fx_right:.3f}")
    print(f"  fy = {fy_right:.3f}")
    print(f"  cx = {cx_right:.3f}")
    print(f"  cy = {cy_right:.3f}\n")

    print(f"Stereo Baseline (tx): {tx:.6f} meters")


def main():
    # >>> ros2 change
    rclpy.init()
    # <<< ros2 end of change


    init = sl.InitParameters()
    init.depth_mode = sl.DEPTH_MODE.NEURAL  # Set configuration parameters for the ZED
    init.async_image_retrieval = False
    # This parameter can be used to record SVO in camera FPS even if  the grab loop is running at a lower FPS (due to compute for ex.)

    status = cam.open(init)

    if status != sl.ERROR_CODE.SUCCESS:
        print("Camera Open", status, "Exit program.")
        exit(1)

    # recording_param = sl.RecordingParameters(opt.output_svo_file, sl.SVO_COMPRESSION_MODE.H265) # Enable recording with the filename specified in argument
    # err = cam.enable_recording(recording_param)
    # if err != sl.ERROR_CODE.SUCCESS:
    #    print("Recording ZED : ", err)
    #    exit(1)

    runtime = sl.RuntimeParameters()
    # print("SVO is Recording, use Ctrl-C to stop.") # Start recording SVO, stop with Ctrl-C command
    frames_recorded = 0

    resolution = cam.get_camera_information().camera_configuration.resolution
    w = min(720, resolution.width)
    h = min(404, resolution.height)
    low_res = sl.Resolution(w, h)

    cam_info = cam.get_camera_information()
    calibration_params = cam_info.camera_configuration.calibration_parameters

    print_params(calibration_params)

    fx = calibration_params.left_cam.fx
    fy = calibration_params.left_cam.fy

    # potentially, need to tune these
    intrinsics = ransac.CameraIntrinsics(w / 2, h / 2, fx / 2, fy / 2)
    drive_conf = ransac.OccupancyGridConfiguration(5000, 5000, 50, thres=5)
    block_conf = ransac.OccupancyGridConfiguration(5000, 5000, 50, thres=1)

    # >>> ros2 change
    grid_width = drive_conf.gw // drive_conf.cw
    grid_height = drive_conf.gh // drive_conf.cw
    # Assuming gw/cw units are millimeters -> convert cell size to meters
    cell_resolution_m = drive_conf.cw / 1000.0
    occ_node = OccGridPublisher(grid_width, grid_height, cell_resolution_m)
    # <<< ros2 end of change


    image_mat = sl.Mat()
    depth_m = sl.Mat()

    key = 0
    while key != 113:  # for 'q' key
        err = cam.grab(runtime)
        if err <= sl.ERROR_CODE.SUCCESS:  # good to go
            # FIXME pointing camera at only the ground causing a crash
            cam.retrieve_image(image_mat, sl.VIEW.LEFT, sl.MEM.CPU, low_res)
            cam.retrieve_measure(depth_m, sl.MEASURE.DEPTH, sl.MEM.CPU, low_res)

            image = image_mat.get_data()
            depths = ransac.plane.clean_depths(depth_m.get_data())

            # ACTUAL USE
            # ransac_output, ransac_coeffs = ransac.plane.hsv_and_ransac(image, depths, 60, (1, 16), 0.15)
            # GROUND ONLY
            ransac_output, ransac_coeffs = ransac.plane.ground_plane(
                depths, 60, (1, 16), 0.15
            )

            rc = ransac.plane.real_coeffs(ransac_coeffs, intrinsics)
            rad = ransac.plane.real_angle(rc)

            drive_ppc = ransac.occu.create_point_cloud(ransac_output, depths)
            drive_rpc = ransac.occu.pixel_to_real(drive_ppc, rc, intrinsics)
            block_ppc = ransac.occu.create_point_cloud(ransac_output != 1, depths)
            block_rpc = ransac.occu.pixel_to_real(block_ppc, rc, intrinsics)

            drive_occ = ransac.occu.occupancy_grid(drive_rpc, drive_conf)
            block_occ = ransac.occu.occupancy_grid(block_rpc, block_conf)
            merged = ransac.occu.merge(drive_occ, block_occ)

            occ_h, occ_w = merged.shape

            #small change here to resolve naming conflict
            virt_cam = ransac.VirtualCamera(occ_h - 1, occ_w // 2, math.pi / 2, math.pi / 2)
            merged = ransac.occu.create_los_grid(merged)  # , [virt_cam])


            # >>> ros2 change
            occ_node.publish(merged)
            # <<< ros2 end of change

            merged = cv2.cvtColor(merged, cv2.COLOR_GRAY2BGR)
            merged = cv2.resize(
                merged, (600, 600), interpolation=cv2.INTER_NEAREST_EXACT
            )
            cv2.imshow("occupancy grid", merged)

            print(f"angle: {math.degrees(rad): .3f} deg")

            key = cv2.waitKey(1)
            # >>> ros2 change
            rclpy.spin_once(occ_node, timeout_sec=0.0)
            # <<< ros2 end of change

        else:
            print("Grab ZED : ", err)
            break
    cv2.destroyAllWindows()
    cam.close()
    # >>> ros2 change
    rclpy.shutdown()
    # <<< ros2 end of change



if __name__ == "__main__":
    main()
