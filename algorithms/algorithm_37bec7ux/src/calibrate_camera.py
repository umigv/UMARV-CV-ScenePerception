"""
Assume I have
    - point clounds, occ grids
    - UI with
        - keyboard input L/R
        - sliders for angle, translation
        - displays merged occ grid
        - display of camera angles / translations.
        - 110 degree FOV
"""

import cv2
import numpy as np
import math
import os
import time
import torch

import pyzed.sl as sl
import ransac_pt as ransac
import ransac_pt.plane
import ransac_pt.occu

# torch.cuda.is_available = lambda: False # force CPU

if torch.cuda.is_available():
    sl_device = sl.MEM.GPU
else:
    sl_device = sl.MEM.CPU


class CameraMergeUI:
    def __init__(self, grid_size=160, panel_size=240):
        self.grid_size = grid_size
        self.panel_size = panel_size

        self.window_name = "Camera Merge Tuner"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1800, 1250)

        self.pause = False

        self._times = []
        self._fps = 0.0
        self._profile_text = ""

    def _draw_centered_text(self, img, text, y, scale=0.45, color=(220, 220, 220)):
        (tw, _), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1
        )
        x = (img.shape[1] - tw) // 2
        cv2.putText(
            img,
            text,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            1,
            cv2.LINE_AA
        )

    def _colorize_grid(self, grid):
        return cv2.applyColorMap((grid).astype(np.uint8), cv2.COLORMAP_BONE)

    def _make_panel(self, title, img, title_color):
        panel = np.ones((self.panel_size, self.panel_size, 3),
                        dtype=np.uint8) * 42

        self._draw_centered_text(
            panel,
            title,
            y=32,
            scale=0.48,
            color=title_color
        )

        img_resized = cv2.resize(
            img, (self.panel_size - 52, self.panel_size - 52))

        h, w = img_resized.shape[:2]
        y0 = (self.panel_size - h) // 2 + 26
        x0 = (self.panel_size - w) // 2
        panel[y0:y0 + h, x0:x0 + w] = img_resized
        return panel

    def _draw_camera_top_view(self, width, angle, displacement, z_offset):
        height = 260
        canvas = np.ones((height, width, 3), dtype=np.uint8) * 30

        fov_half = np.deg2rad(110 / 2)
        center_y = height // 2 + 10
        baseline = displacement * 2
        z_shift = z_offset * 2

        cam1 = np.array([width // 2 - baseline, center_y + z_shift])
        cam2 = np.array([width // 2 + baseline, center_y - z_shift])

        yaw = np.deg2rad(angle / 2)

        def draw_camera(center, yaw_angle, color):
            forward = -np.pi / 2 + yaw_angle
            for a in (forward - fov_half, forward + fov_half):
                end = center + 110 * np.array([math.cos(a), math.sin(a)])
                cv2.line(canvas, center.astype(int), end.astype(int), color, 2)
            cv2.circle(canvas, tuple(center.astype(int)), 5, color, -1)

        draw_camera(cam1, -yaw, (255, 0, 0))
        draw_camera(cam2, +yaw, (0, 255, 0))

        cv2.putText(
            canvas,
            "Camera Geometry (Top View)",
            (30, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (220, 220, 220),
            1
        )

        return canvas

    def _draw_score_visual(self, width, current_score, best_score, best_params=None):
        height = 260
        canvas = np.ones((height, width, 3), dtype=np.uint8) * 30

        cv2.putText(
            canvas,
            "Match Metrics",
            (20, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (220, 220, 220),
            1
        )

        def draw_bar(label, value, max_val, y, color):
            cv2.putText(
                canvas,
                f"{label}: {value}",
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (200, 200, 200),
                1
            )
            bar_w = int((width - 40) * (value / max_val)) if max_val > 0 else 0
            cv2.rectangle(canvas, (20, y + 10),
                          (20 + bar_w, y + 25), color, -1)
            cv2.rectangle(canvas, (20, y + 10),
                          (width - 20, y + 25), (60, 60, 60), 1)

        max_possible = self.grid_size * self.grid_size
        draw_bar("Current Score", current_score,
                 max_possible, 75, (120, 255, 120))
        draw_bar("Best Session Score", best_score,
                 max_possible, 140, (255, 200, 100))

        if best_params:
            angle, disp, z_off = best_params
            best_text = f"A:{angle} D:{disp} Z:{z_off}"
            cv2.putText(
                canvas,
                best_text,
                (20, 200),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (180, 180, 180),
                1
            )

        return canvas

    def _draw_title_bar(self, width):
        bar = np.ones((60, width, 3), dtype=np.uint8) * 26

        self._draw_centered_text(
            bar,
            "Camera Merge Tuner",
            y=40,
            scale=1.05,
            color=(245, 245, 245)
        )

        fps_text = f"{self._fps:.1f} FPS"
        (tw, _), _ = cv2.getTextSize(
            fps_text,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            1
        )

        cv2.putText(
            bar,
            fps_text,
            (width - tw - 20, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (180, 220, 255),
            1,
            cv2.LINE_AA
        )

        return bar

    def _draw_controls_bar(self, width, angle, displacement, z_offset):
        h = 80
        bar = np.ones((h, width, 3), dtype=np.uint8) * 28

        values_text = f"Angle: {angle} deg    Disp: {displacement} cm    Z-Off: {z_offset} cm"
        cv2.putText(
            bar,
            values_text,
            (20, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (220, 220, 220),
            1
        )

        controls_text = "Q/E: Angle | A/D: Disp | W/S: Z-Off | P: Pause | X: Exit | M: Save | N: Save Best"
        cv2.putText(
            bar,
            controls_text,
            (20, 64),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (180, 180, 180),
            1
        )

        return bar

    def _draw_profile_bar(self, width):
        h = 30
        bar = np.ones((h, width, 3), dtype=np.uint8) * 26

        cv2.putText(
            bar,
            self._profile_text,
            (20, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (200, 200, 200),
            1,
            cv2.LINE_AA
        )
        return bar

    def render(
        self,
        occ1,
        occ2,
        merged_occ,
        angle,
        displacement,
        z_offset,
        current_score=0,
        best_score=0,
        best_params=None,
        profile_text=""
    ):

        now = time.time()
        self._profile_text = profile_text
        self._times.append(now)

        if len(self._times) > 60:
            self._times = self._times[-60:]

        if len(self._times) > 1:
            self._fps = (len(self._times) - 1) / \
                (self._times[-1] - self._times[0])
        else:
            self._fps = 0

        occ_row = np.hstack([
            self._make_panel("Camera 1 - Occ",
                             self._colorize_grid(occ1), (255, 120, 120)),
            self._make_panel("Camera 2 - Occ",
                             self._colorize_grid(occ2), (120, 255, 120)),
            self._make_panel(
                "Merged - Occ", self._colorize_grid(merged_occ), (220, 220, 220)),
        ])

        width = occ_row.shape[1]

        title_bar = self._draw_title_bar(width)

        cam_view_w = int(width * 0.7)
        score_view_w = width - cam_view_w
        cam_top_view = self._draw_camera_top_view(
            cam_view_w, angle, displacement, z_offset)
        score_view = self._draw_score_visual(
            score_view_w, current_score, best_score, best_params)
        camera_view = np.hstack([cam_top_view, score_view])

        controls = self._draw_controls_bar(
            width, angle, displacement, z_offset)
        profile_bar = self._draw_profile_bar(width)

        dashboard = np.vstack([
            title_bar,
            occ_row,
            camera_view,
            controls,
            profile_bar
        ])

        cv2.imshow(self.window_name, dashboard)

    def handle_keyboard(self, angle, displacement, z_offset):

        key = cv2.waitKey(30) & 0xFF

        if key in (ord('q'), ord('Q')):
            angle = min(90, angle + 1)

        elif key in (ord('e'), ord('E')):
            angle = max(-90, angle - 1)

        elif key in (ord('w'), ord('W')):
            z_offset += 1

        elif key in (ord('s'), ord('S')):
            z_offset -= 1

        elif key in (ord('a'), ord('A')):
            displacement -= 1

        elif key in (ord('d'), ord('D')):
            displacement += 1

        elif key in (ord('p'), ord('P')):
            self.pause = not self.pause

        return key, angle, displacement, z_offset


cam = sl.Camera()


def main():
    cams = []
    init = sl.InitParameters()
    init.depth_mode = sl.DEPTH_MODE.NEURAL
    init.async_image_retrieval = False

    devices = sl.Camera.get_device_list()
    if len(devices) < 2:
        print("Need at least 2 ZED cameras.")
        exit(1)

    for dev in devices[:2]:
        cam = sl.Camera()
        init.set_from_serial_number(dev.serial_number)
        status = cam.open(init)
        if status != sl.ERROR_CODE.SUCCESS:
            print("Camera open failed:", status)
            exit(1)
        cams.append(cam)

    runtime = sl.RuntimeParameters()

    intr = [None, None]
    for i in range(2):
        cam_info = cams[i].get_camera_information()
        resolution = cam_info.camera_configuration.resolution
        w = min(720, resolution.width)
        h = min(404, resolution.height)
        low_res = sl.Resolution(w, h)

        calibration_params = cam_info.camera_configuration.calibration_parameters
        fx = calibration_params.left_cam.fx
        fy = calibration_params.left_cam.fy

        intr[i] = ransac.Intrinsics(w / 2, h / 2, fx / 2, fy / 2)

    drive_conf = ransac.GridConfiguration(5000, 5000, 50, thres=2)
    block_conf = ransac.GridConfiguration(5000, 5000, 50, thres=1)

    image_mats = [sl.Mat(), sl.Mat()]
    depth_mats = [sl.Mat(), sl.Mat()]

    ui = CameraMergeUI(grid_size=60)
    angle_deg = 0
    displacement_cm = 0
    z_offset_cm = 0

    key = 0

    last_drive_ppc = [None, None]
    last_block_ppc = [None, None]
    last_real_coeffs = [None, None]

    px_coeffs_cache = [np.array([0, 0, 0]), np.array([0, 0, 0])]

    best_score = 0
    best_params = (0, 0, 0)
    was_paused = False

    while True:
        frame_start = time.perf_counter()

        t_grab = 0.0
        t_ransac = 0.0
        t_occ = 0.0
        t_merge = 0.0

        occ_grids = []

        for i in range(2):

            if not ui.pause:
                t0 = time.perf_counter()

                err = cams[i].grab(runtime)
                if err != sl.ERROR_CODE.SUCCESS:
                    print("Grab error:", err)
                    continue

                cams[i].retrieve_image(
                    image_mats[i], sl.VIEW.LEFT, sl.MEM.GPU, low_res)
                cams[i].retrieve_measure(
                    depth_mats[i], sl.MEASURE.DEPTH, sl.MEM.GPU, low_res)

                t_grab += (time.perf_counter() - t0)

                image = image_mats[i].get_data(sl.MEM.GPU)
                depths = ransac.plane.clean_depths(
                    depth_mats[i].get_data(sl.MEM.GPU))

                t0 = time.perf_counter()

                ransac_output, px_coeffs_cache[i] = ransac.plane.ground_plane(
                    depths,
                    60,
                    (1, 16),
                    0.15,
                    guess=px_coeffs_cache[i]
                )

                real_coeffs = ransac.plane.real_coeffs(
                    px_coeffs_cache[i], intr[i])

                t_ransac += (time.perf_counter() - t0)

                drive_ppc = ransac.occu.create_point_cloud(
                    ransac_output, depths)
                block_ppc = ransac.occu.create_point_cloud(
                    ransac_output != 1, depths
                )

                last_drive_ppc[i] = drive_ppc
                last_block_ppc[i] = block_ppc
                last_real_coeffs[i] = real_coeffs

            else:
                if last_drive_ppc[i] is None or last_block_ppc[i] is None or last_real_coeffs[i] is None:
                    t0 = time.perf_counter()

                    err = cams[i].grab(runtime)
                    if err != sl.ERROR_CODE.SUCCESS:
                        print("Grab error during pause-init:", err)
                        continue

                    cams[i].retrieve_image(
                        image_mats[i], sl.VIEW.LEFT, sl.MEM.CPU, low_res)
                    cams[i].retrieve_measure(
                        depth_mats[i], sl.MEASURE.DEPTH, sl.MEM.CPU, low_res)

                    t_grab += (time.perf_counter() - t0)

                    image = image_mats[i].get_data()
                    depths = ransac.plane.clean_depths(
                        depth_mats[i].get_data())

                    t0 = time.perf_counter()

                    ransac_output, px_coeffs_cache[i] = ransac.plane.ground_plane(
                        depths,
                        60,
                        (1, 16),
                        0.15,
                        guess=px_coeffs_cache[i]
                    )

                    last_real_coeffs[i] = ransac.plane.real_coeffs(
                        px_coeffs_cache[i], intr[i])

                    t_ransac += (time.perf_counter() - t0)

                    last_drive_ppc[i] = ransac.occu.create_point_cloud(
                        ransac_output, depths)
                    last_block_ppc[i] = ransac.occu.create_point_cloud(
                        ransac_output != 1, depths
                    )

            if last_drive_ppc[i] is None or last_block_ppc[i] is None or last_real_coeffs[i] is None:
                continue

            half_angle_rad = np.deg2rad(angle_deg / 2)
            half_displacement_mm = displacement_cm * 10 / 2
            half_z_offset_mm = z_offset_cm * 10 / 2

            t0 = time.perf_counter()

            drive_rpc = ransac.occu.pixel_to_real(
                last_drive_ppc[i],
                last_real_coeffs[i],
                intr[i],
                half_angle_rad * (1 if i == 0 else -1)
            )
            drive_rpc[:, 0] += (-1 if i == 0 else 1) * half_displacement_mm
            drive_rpc[:, 2] += (-1 if i == 0 else 1) * half_z_offset_mm

            block_rpc = ransac.occu.pixel_to_real(
                last_block_ppc[i],
                last_real_coeffs[i],
                intr[i],
                half_angle_rad * (1 if i == 0 else -1)
            )
            block_rpc[:, 0] += (-1 if i == 0 else 1) * half_displacement_mm
            block_rpc[:, 2] += (-1 if i == 0 else 1) * half_z_offset_mm

            drive_occ = ransac.occu.occupancy_grid(drive_rpc, drive_conf)
            block_occ = ransac.occu.occupancy_grid(block_rpc, block_conf)

            full_occ = ransac.occu.composite(drive_occ, block_occ)

            t_occ += (time.perf_counter() - t0)

            occ_grids.append(full_occ)

        if len(occ_grids) < 2:
            continue

        occ1 = occ_grids[0]
        occ2 = occ_grids[1]

        t0 = time.perf_counter()

        merged_occ = (occ1.astype(np.int32) + occ2.astype(np.int32)) // 2
        merged_occ = np.where(
            merged_occ > 127,
            255,
            np.where(merged_occ < 127, 0, 127)
        ).astype(np.uint8)

        t_merge += (time.perf_counter() - t0)

        if ui.pause:
            if not was_paused:
                best_score = 0
                was_paused = True
            current_score = int(np.sum((occ1 == 255) & (
                occ2 == 255)) + np.sum((occ1 == 0) & (occ2 == 0)))
            if current_score > best_score:
                best_score = current_score
                best_params = (angle_deg, displacement_cm, z_offset_cm)
        else:
            was_paused = False
            current_score = 0
            best_score = 0

        total_time = time.perf_counter() - frame_start

        profile_text = (
            f"Grab: {t_grab*1000:5.1f}ms | "
            f"RANSAC: {t_ransac*1000:5.1f}ms | "
            f"Occ: {t_occ*1000:5.1f}ms | "
            f"Merge: {t_merge*1000:5.1f}ms | "
            f"Total: {total_time*1000:5.1f}ms"
        )

        ui.render(
            occ1=occ1,
            occ2=occ2,
            merged_occ=merged_occ,
            angle=angle_deg,
            displacement=displacement_cm,
            z_offset=z_offset_cm,
            current_score=current_score,
            best_score=best_score,
            best_params=best_params if ui.pause else None,
            profile_text=profile_text
        )

        key, angle_deg, displacement_cm, z_offset_cm = ui.handle_keyboard(
            angle_deg,
            displacement_cm,
            z_offset_cm
        )

        if key in (27, ord('x'), ord('X')):
            break

        if key in (ord('m'), ord('M')):
            os.makedirs("saves/cam_calibration", exist_ok=True)

            if os.listdir("saves/cam_calibration"):
                for f in os.listdir("saves/cam_calibration"):
                    os.remove(os.path.join("saves/cam_calibration", f))

            np.savez(
                f"saves/cam_calibration/angle_{angle_deg}_disp_{displacement_cm}_zoff_{z_offset_cm}.npz",
                angle=angle_deg,
                displacement=displacement_cm,
                z_offset=z_offset_cm
            )
            print(
                f"Saved calibration: angle={angle_deg}, disp={displacement_cm}, z_offset={z_offset_cm}")

            break

        if key in (ord('n'), ord('N')):
            os.makedirs("saves/cam_calibration", exist_ok=True)

            if os.listdir("saves/cam_calibration"):
                for f in os.listdir("saves/cam_calibration"):
                    os.remove(os.path.join("saves/cam_calibration", f))

            ba, bd, bz = best_params
            np.savez(
                f"saves/cam_calibration/best_angle_{ba}_disp_{bd}_zoff_{bz}.npz",
                angle=ba,
                displacement=bd,
                z_offset=bz
            )
            print(
                f"Saved BEST calibration: angle={ba}, disp={bd}, z_offset={bz}")

            break

    for cam in cams:
        cam.close()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
