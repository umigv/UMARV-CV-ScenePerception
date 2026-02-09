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
import ransac.plane
import ransac.occu
import numpy as np
import math
from calibrate_camera import CameraMergeUI

cam = sl.Camera()


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
    global cam

    init = sl.InitParameters()
    # Set configuration parameters for the ZED
    init.depth_mode = sl.DEPTH_MODE.NEURAL
    init.async_image_retrieval = False
    # This parameter can be used to record SVO in camera FPS even if the grab loop is running at a lower FPS (due to compute for ex.)

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
    intr = ransac.Intrinsics(w / 2, h / 2, fx / 2, fy / 2)
    drive_conf = ransac.GridConfiguration(5000, 5000, 50, thres=2)
    block_conf = ransac.GridConfiguration(5000, 5000, 50, thres=1)

    image_mat = sl.Mat()
    depth_m = sl.Mat()

    ui = CameraMergeUI(grid_size=60)
    angle_deg, displacement_cm = 0, 0 # camera centered, facing forward.

    key = 0
    while key != 113:  # for 'q' key
        err = cam.grab(runtime)
        if err <= sl.ERROR_CODE.SUCCESS:  # good to go
            # FIXME pointing camera at only the ground causing a crash
            cam.retrieve_image(image_mat, sl.VIEW.LEFT, sl.MEM.CPU, low_res)
            cam.retrieve_measure(
                depth_m, sl.MEASURE.DEPTH, sl.MEM.CPU, low_res)

            image = image_mat.get_data()
            depths = ransac.plane.clean_depths(depth_m.get_data())

            # ACTUAL USE
            # ransac_output, ransac_coeffs = ransac.plane.hsv_and_ransac(image, depths, 60, (1, 16), 0.15)
            # GROUND ONLY
            ransac_output, px_coeffs = ransac.plane.ground_plane(
                depths, 60, (1, 16), 0.15
            )

            real_coeffs = ransac.plane.real_coeffs(px_coeffs, intr)
            rad = ransac.plane.real_angle(real_coeffs)

            drive_ppc = ransac.occu.create_point_cloud(ransac_output, depths)
            drive_rpc = ransac.occu.pixel_to_real(
                drive_ppc, real_coeffs, intr)
            block_ppc = ransac.occu.create_point_cloud(
                ransac_output != 1, depths)
            block_rpc = ransac.occu.pixel_to_real(
                block_ppc, real_coeffs, intr)

            drive_occ = ransac.occu.occupancy_grid(drive_rpc, drive_conf)
            block_occ = ransac.occu.occupancy_grid(block_rpc, block_conf)
            full_occ = ransac.occu.composite(drive_occ, block_occ)

            x_offset_drive = np.ones(drive_rpc.shape[0]) * (displacement_cm / 10)
            x_offset_block = np.ones(block_rpc.shape[0]) * (displacement_cm / 10)

            drive_rpc_2d = np.delete(drive_rpc, (1), axis=1)
            drive_rpc_2d[:, 0] = drive_rpc_2d[:, 0] + x_offset_drive

            block_rpc_2d = np.delete(block_rpc, (1), axis=1)
            block_rpc_2d[:, 0] = block_rpc_2d[:, 0] + x_offset_block

            # Mock right camera view: replace full_occ_right with real right camera occ grid
            mock_displacement = 12
            mock_angle = 37
            transform = cv2.getRotationMatrix2D((full_occ.shape[1]//2, full_occ.shape[0]//2), mock_angle, 1)
            transform[0, 2] -= mock_displacement

            full_occ_right = full_occ.copy()            
            full_occ_right = cv2.warpAffine(full_occ_right, transform, (full_occ.shape[1], full_occ.shape[0]), flags=cv2.INTER_LINEAR)
            full_occ_left = full_occ.copy()
            # # #


            transform_left = cv2.getRotationMatrix2D((full_occ.shape[1]//2, full_occ.shape[0]//2), angle_deg/2, 1)
            transform_left[0, 2] -= displacement_cm/2
            transform_right = cv2.getRotationMatrix2D((full_occ.shape[1]//2, full_occ.shape[0]//2), -angle_deg/2, 1)
            transform_right[0, 2] += displacement_cm/2

            occ1 = cv2.warpAffine(full_occ_left, transform_left, (full_occ.shape[1], full_occ.shape[0]), flags=cv2.INTER_LINEAR)
            occ2 = cv2.warpAffine(full_occ_right, transform_right, (full_occ.shape[1], full_occ.shape[0]), flags=cv2.INTER_LINEAR)
            
            merged_occ = np.maximum(occ1, occ2)
            merged_occ = np.where((occ1 == 128) | (occ2 == 128), 
                                np.maximum(occ1, occ2), 
                                merged_occ)

            ui.render(
                occ1=occ1,
                occ2=occ2,
                merged_occ=merged_occ,
                angle=angle_deg,
                displacement=displacement_cm
            )

            key, angle_deg, displacement_cm = ui.handle_keyboard(angle_deg, displacement_cm)

            if key in (27, ord('q')):
                break
            
            # --- #

            occ_h, occ_w = full_occ.shape
            vcam = ransac.VirtualCamera(
                occ_h - 1, occ_w // 2, math.pi / 2, math.radians(110))
            full_occ = ransac.occu.create_los_grid(full_occ, [vcam])

            full_occ = cv2.cvtColor(full_occ, cv2.COLOR_GRAY2BGR)
            full_occ = cv2.resize(
                full_occ, (600, 600), interpolation=cv2.INTER_NEAREST_EXACT
            )
            cv2.imshow("occupancy grid", full_occ)

            x = w // 2
            y = h // 2
            # coords = None # disables conversion
            coords = np.array([[x, y]])  # pixel coordinates
            pred_real = np.array([])
            if coords is not None:
                pred = ransac.occu.create_ground_cloud(
                    coords, px_coeffs)
                # n by 2 array of (x, z) coordinates
                pred_real = ransac.occu.pixel_to_real(
                    pred, real_coeffs, intr)[:, (0, 2)]
            print(pred_real)

            print(f"angle: {math.degrees(rad): .3f} deg")

            key = cv2.waitKey(1)
        else:
            print("Grab ZED : ", err)
            break
    cv2.destroyAllWindows()
    cam.close()


if __name__ == "__main__":
    main()





            