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
import ransac.occu, ransac.plane

cam = sl.Camera()


# Handler to deal with CTRL+C properly
def handler(signal_received, frame):
    cam.disable_recording()
    cam.close()
    sys.exit(0)


signal(SIGINT, handler)


def main():

    init = sl.InitParameters()
    init.depth_mode = sl.DEPTH_MODE.NEURAL  # Set configuration parameters for the ZED
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
    low_resolution = sl.Resolution(w * 2, h)
    low_resolution_d = sl.Resolution(w, h)

    cam_info = cam.get_camera_information()
    calibration_params = cam_info.camera_configuration.calibration_parameters

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

    # potentially, need to tune these
    intrinsics = ransac.CameraIntrinsics(w / 2, h / 2, fx_left / 2, fy_left / 2)

    image = sl.Mat()
    depth = sl.Mat()

    key = 0
    while key != 113:  # for 'q' key
        err = cam.grab(runtime)
        if err <= sl.ERROR_CODE.SUCCESS:  # good to go
            cam.retrieve_image(
                image, sl.VIEW.SIDE_BY_SIDE, sl.MEM.CPU, low_resolution
            )  # retrieve image left and right
            cam.retrieve_measure(depth, sl.MEASURE.DEPTH, sl.MEM.CPU, low_resolution_d)
            # print(svo_image)
            # print(depth_map)
            # print(depth_map.get_data())
            # todo: deal with nan and infinity

            img_arr = image.get_data()[:, :, :3]
            depth_arr = ransac.plane.clean_depths(depth.get_data())

            ransac_output, ransac_coeffs = ransac.plane.ransac(
                depth_arr, 60, (1, 16), 0.1
            )
            ransac_output = ransac_output[100:, :]

            real = ransac.plane.real_coeffs(ransac_coeffs, intrinsics)
            angle = ransac.plane.real_angle(real)

            print(angle)

            svo_position = cam.get_svo_position()

            cv2.imshow("View", img_arr)  # dislay both images to cv
            key = cv2.waitKey(1)
            # if key == 115 :# for 's' key
            #     #save .svo image as a png
            #     cam.retrieve_image(mat)
            #     filepath = "capture_" + str(svo_position) + ".png"
            #     img = mat.write(filepath)
            #     if img == sl.ERROR_CODE.SUCCESS:
            #         print("Saved image : ",filepath)
            #     else:
            #         print("Something wrong happened in image saving... ")
        else:
            print("Grab ZED : ", err)
            break
    cv2.destroyAllWindows()
    cam.close()


if __name__ == "__main__":
    main()
