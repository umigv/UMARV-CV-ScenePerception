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
import glob
import ransac_pt as ransac
import ransac_pt.plane
import ransac_pt.occu
import numpy as np
import math

cam = sl.Camera()
cams = []


# Handler to deal with CTRL+C properly
def handler(signal_received, frame):
    try:
        for c in cams:
            try:
                c.disable_recording()
                c.close()
            except Exception:
                pass
    except Exception:
        try:
            cam.disable_recording()
            cam.close()
        except Exception:
            pass
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

    devices = sl.Camera.get_device_list()
    if len(devices) >= 2:
        for dev in devices[:2]:
            c = sl.Camera()
            init.set_from_serial_number(dev.serial_number)
            status = c.open(init)
            if status != sl.ERROR_CODE.SUCCESS:
                print("Camera open failed:", status)
                exit(1)
            cams.append(c)
        cam = cams[0]
        print(f"Opened {len(cams)} cameras")
    else:
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

    cam_info = cam.get_camera_information()
    resolution = cam_info.camera_configuration.resolution
    w = min(720, resolution.width)
    h = min(404, resolution.height)

    low_res = sl.Resolution(w, h)

    calibration_params = cam_info.camera_configuration.calibration_parameters

    print_params(calibration_params)

    fx = calibration_params.left_cam.fx
    fy = calibration_params.left_cam.fy

    # potentially, need to tune these
    intr = ransac.Intrinsics(w / 2, h / 2, fx / 2, fy / 2)
    drive_conf = ransac.GridConfiguration(5000, 5000, 50, thres=2)
    block_conf = ransac.GridConfiguration(5000, 5000, 50, thres=1)

    image_mats = [sl.Mat(), sl.Mat()]
    depth_mats = [sl.Mat(), sl.Mat()]

    angle_deg, displacement_cm, z_offset_cm = 0, 0, 0

    calib_dir = "saves/cam_calibration"
    if os.path.isdir(calib_dir):
        files = glob.glob(os.path.join(calib_dir, "*.npz"))
        if files:
            latest = max(files, key=os.path.getmtime)
            data = np.load(latest)
            if "angle" in data:
                angle_deg = float(data["angle"].tolist())
            if "displacement" in data:
                displacement_cm = float(data["displacement"].tolist())
            if "z_offset" in data:
                z_offset_cm = float(data["z_offset"].tolist())
            print(f"Loaded calibration from {latest}")

    half_angle_rad = np.deg2rad(angle_deg / 2)
    half_displacement_mm = displacement_cm * 10 / 2
    half_z_offset_mm = z_offset_cm * 10 / 2

    px_coeffs_cache = [np.array([0, 0, 0]), np.array([0, 0, 0])]
    real_coeffs_cache = [None, None]

    key = 0
    while key != 113:  # for 'q' key
        for i in range(2):
            cams[i].grab(runtime)
            cams[i].retrieve_image(image_mats[i], sl.VIEW.LEFT, sl.MEM.GPU, low_res)
            cams[i].retrieve_measure(depth_mats[i], sl.MEASURE.DEPTH, sl.MEM.GPU, low_res)

        occ_grids = []
        px_coeffs_left = None
        real_coeffs_left = None
        for i in range(2):
            image = image_mats[i].get_data(sl.MEM.GPU)
            depths = ransac.plane.clean_depths(depth_mats[i].get_data(sl.MEM.GPU))

            ransac_output, px_coeffs_cache[i] = ransac.plane.ground_plane(
                depths,
                60,
                (1, 16),
                0.15,
                guess=px_coeffs_cache[i]
            )

            real_coeffs_cache[i] = ransac.plane.real_coeffs(px_coeffs_cache[i], intr)
            real_coeffs = real_coeffs_cache[i]

            if i == 0:
                px_coeffs_left = px_coeffs_cache[i]
                real_coeffs_left = real_coeffs_cache[i]

            sign = 1 if i == 0 else -1

            drive_ppc = ransac.occu.create_point_cloud(ransac_output, depths)
            drive_rpc = ransac.occu.pixel_to_real(drive_ppc, real_coeffs, intr, half_angle_rad * sign)
            drive_rpc[:, 0] += -sign * half_displacement_mm
            drive_rpc[:, 2] += -sign * half_z_offset_mm

            block_ppc = ransac.occu.create_point_cloud(ransac_output != 1, depths)
            block_rpc = ransac.occu.pixel_to_real(block_ppc, real_coeffs, intr, half_angle_rad * sign)
            block_rpc[:, 0] += -sign * half_displacement_mm
            block_rpc[:, 2] += -sign * half_z_offset_mm

            drive_occ = ransac.occu.occupancy_grid(drive_rpc, drive_conf)
            block_occ = ransac.occu.occupancy_grid(block_rpc, block_conf)

            full_occ = ransac.occu.composite(drive_occ, block_occ)

            occ_grids.append(full_occ)

        occ1 = occ_grids[0]
        occ2 = occ_grids[1]
    
        merged_occ = (occ1.astype(np.int32) + occ2.astype(np.int32)) // 2
        merged_occ = np.where(
            merged_occ > 127,
            255,
            np.where(merged_occ < 127, 0, 127)
        ).astype(np.uint8)

        cv2.imshow("occ1", cv2.resize(occ1, (600, 600), interpolation=cv2.INTER_NEAREST_EXACT))
        cv2.imshow("occ2", cv2.resize(occ2, (600, 600), interpolation=cv2.INTER_NEAREST_EXACT))
        cv2.imshow("merged_occ", cv2.resize(merged_occ, (600, 600), interpolation=cv2.INTER_NEAREST_EXACT))

        occ_h, occ_w = occ1.shape
        vcam = ransac.VirtualCamera(occ_h - 1, occ_w // 2, math.pi / 2, math.radians(110))
        los = ransac.occu.create_los_grid(merged_occ, [vcam])
        los = cv2.cvtColor(los, cv2.COLOR_GRAY2BGR)
        los = cv2.resize(los, (600, 600), interpolation=cv2.INTER_NEAREST_EXACT)
        cv2.imshow("occupancy grid", los)

        x = w // 2
        y = h // 2
        coords = np.array([[x, y]])
        pred_real = np.array([])
        if px_coeffs_left is not None:
            pred = ransac.occu.create_ground_cloud(coords, px_coeffs_left)
            pred_real = ransac.occu.pixel_to_real(pred, real_coeffs_left, intr)[:, (0, 2)]
        print(pred_real)

        try:
            rad = ransac.plane.real_angle(real_coeffs_left)
            print(f"angle: {math.degrees(rad): .3f} deg")
        except Exception:
            pass

        key = cv2.waitKey(1)
    cv2.destroyAllWindows()
    cam.close()


if __name__ == "__main__":
    main()





            