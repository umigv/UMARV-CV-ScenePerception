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

import pyzed.sl as sl
import ransac.plane
import ransac.occu




class CameraMergeUI:
    def __init__(self, grid_size=160, panel_size=240):
        self.grid_size = grid_size
        self.panel_size = panel_size

        self.window_name = "Camera Merge Tuner"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1800, 1250)

        self.pause = False

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
        panel = np.ones((self.panel_size, self.panel_size, 3), dtype=np.uint8) * 42

        self._draw_centered_text(
            panel,
            title,
            y=32,
            scale=0.48,
            color=title_color
        )

        # Resize image to fit panel
        img_resized = cv2.resize(img, (self.panel_size - 52, self.panel_size - 52))
        
        h, w = img_resized.shape[:2]
        y0 = (self.panel_size - h) // 2 + 26
        x0 = (self.panel_size - w) // 2
        panel[y0:y0 + h, x0:x0 + w] = img_resized
        return panel

    def _draw_camera_top_view(self, width, angle, displacement):
        height = 260
        canvas = np.ones((height, width, 3), dtype=np.uint8) * 30

        fov_half = np.deg2rad(110 / 2)
        center_y = height // 2 + 10
        baseline = displacement * 4

        cam1 = np.array([width // 2 - baseline, center_y])
        cam2 = np.array([width // 2 + baseline, center_y])

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

    def _draw_title_bar(self, width):
        bar = np.ones((60, width, 3), dtype=np.uint8) * 26

        self._draw_centered_text(
            bar,
            "Camera Merge Tuner",
            y=40,
            scale=1.05,
            color=(245, 245, 245)
        )

        return bar

    def _draw_controls_bar(self, width, angle, displacement):
        h = 50
        bar = np.ones((h, width, 3), dtype=np.uint8) * 28

        left_text = f"Angle: {angle} deg    Disp: {displacement} cm"
        cv2.putText(
            bar,
            left_text,
            (20, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (220, 220, 220),
            1
        )

        right_text = "W/S: Angle | A/D: Disp | P: Pause | Q: Quit | X: Save"
        (tw, _), _ = cv2.getTextSize(
            right_text,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            1
        )
        x = width - tw - 20

        cv2.putText(
            bar,
            right_text,
            (x, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (180, 180, 180),
            1
        )

        return bar

    def render(
        self,
        occ1,
        occ2,
        merged_occ,
        angle,
        displacement,
    ):
        occ_row = np.hstack([
            self._make_panel("Camera 1 - Occ", self._colorize_grid(occ1), (255, 120, 120)),
            self._make_panel("Camera 2 - Occ", self._colorize_grid(occ2), (120, 255, 120)),
            self._make_panel("Merged - Occ", self._colorize_grid(merged_occ), (220, 220, 220)),
        ])

        width = occ_row.shape[1]

        title_bar = self._draw_title_bar(width)
        camera_view = self._draw_camera_top_view(width, angle, displacement)
        controls = self._draw_controls_bar(width, angle, displacement)

        dashboard = np.vstack([
            title_bar,
            occ_row,
            camera_view,
            controls
        ])

        cv2.imshow(self.window_name, dashboard)

    
    def handle_keyboard(self, angle, displacement):

        key = cv2.waitKey(30) & 0xFF

        if key in (ord('w'), ord('W')):
            angle = min(90, angle + 1)

        elif key in (ord('s'), ord('S')):
            angle = max(-90, angle - 1)

        elif key in (ord('a'), ord('A')):
            displacement = max(0, displacement - 1)

        elif key in (ord('d'), ord('D')):
            displacement += 1

        elif key in (ord('p'), ord('P')):
            self.pause = not self.pause

        return key, angle, displacement
    


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

    key = 0
    # Keep last raw point-clouds + plane coeffs so pause doesn't grab new frames
    last_drive_ppc = [None, None]
    last_block_ppc = [None, None]
    last_real_coeffs = [None, None]

    while key != 113:

        occ_grids = []

        for i in range(2):

            if not ui.pause:
                err = cams[i].grab(runtime)
                if err != sl.ERROR_CODE.SUCCESS:
                    print("Grab error:", err)
                    continue

                cams[i].retrieve_image(image_mats[i], sl.VIEW.LEFT, sl.MEM.CPU, low_res)
                cams[i].retrieve_measure(depth_mats[i], sl.MEASURE.DEPTH, sl.MEM.CPU, low_res)

                image = image_mats[i].get_data()
                depths = ransac.plane.clean_depths(depth_mats[i].get_data())

                ransac_output, px_coeffs = ransac.plane.ground_plane(
                    depths, 60, (1, 16), 0.15
                )

                real_coeffs = ransac.plane.real_coeffs(px_coeffs, intr[i])

                drive_ppc = ransac.occu.create_point_cloud(ransac_output, depths)

                block_ppc = ransac.occu.create_point_cloud(
                    ransac_output != 1, depths
                )

                # store raw intermediate data so we can reuse while paused
                last_drive_ppc[i] = drive_ppc
                last_block_ppc[i] = block_ppc
                last_real_coeffs[i] = real_coeffs

            else:
                # Paused: reuse last captured raw data; if missing, perform one grab to initialize
                if last_drive_ppc[i] is None or last_block_ppc[i] is None or last_real_coeffs[i] is None:
                    err = cams[i].grab(runtime)
                    if err != sl.ERROR_CODE.SUCCESS:
                        print("Grab error during pause-init:", err)
                        continue
                    cams[i].retrieve_image(image_mats[i], sl.VIEW.LEFT, sl.MEM.CPU, low_res)
                    cams[i].retrieve_measure(depth_mats[i], sl.MEASURE.DEPTH, sl.MEM.CPU, low_res)
                    image = image_mats[i].get_data()
                    depths = ransac.plane.clean_depths(depth_mats[i].get_data())
                    ransac_output, px_coeffs = ransac.plane.ground_plane(
                        depths, 60, (1, 16), 0.15
                    )
                    last_real_coeffs[i] = ransac.plane.real_coeffs(px_coeffs, intr[i])
                    last_drive_ppc[i] = ransac.occu.create_point_cloud(ransac_output, depths)
                    last_block_ppc[i] = ransac.occu.create_point_cloud(
                        ransac_output != 1, depths
                    )

            # If we have stored raw data, compute RPCs and occupancy using current angle/displacement
            if last_drive_ppc[i] is None or last_block_ppc[i] is None or last_real_coeffs[i] is None:
                continue

            half_angle_rad = np.deg2rad(angle_deg / 2)
            half_displacement_mm = displacement_cm * 10 / 2

            drive_rpc = ransac.occu.pixel_to_real(last_drive_ppc[i], last_real_coeffs[i], intr[i], half_angle_rad * (1 if i == 0 else -1))
            drive_rpc[:, 0] += (-1 if i == 0 else 1) * half_displacement_mm
        
            block_rpc = ransac.occu.pixel_to_real(last_block_ppc[i], last_real_coeffs[i], intr[i], half_angle_rad * (1 if i == 0 else -1))
            block_rpc[:, 0] += (-1 if i == 0 else 1) * half_displacement_mm

            drive_occ = ransac.occu.occupancy_grid(drive_rpc, drive_conf)
            block_occ = ransac.occu.occupancy_grid(block_rpc, block_conf)

            full_occ = ransac.occu.composite(drive_occ, block_occ)

            occ_grids.append(full_occ)

        if len(occ_grids) < 2:
            continue

        occ1 = occ_grids[0]
        occ2 = occ_grids[1]

        # Average (int) + threshold
        merged_occ = (occ1.astype(np.int32) + occ2.astype(np.int32)) // 2
        merged_occ = np.where(merged_occ > 127, 255, np.where(merged_occ < 127, 0, 127)).astype(np.uint8)

        ui.render(
            occ1=occ1,
            occ2=occ2,
            merged_occ=merged_occ,
            angle=angle_deg,
            displacement=displacement_cm
        )

        key, angle_deg, displacement_cm = ui.handle_keyboard(
            angle_deg,
            displacement_cm
        )

        if key in (27, ord('q')):
            break

        if key in (ord('x'), ord('X')):
            
            os.makedirs("saves/cam_calibration", exist_ok=True)
            
            if os.listdir("saves/cam_calibration"):
                for f in os.listdir("saves/cam_calibration"):
                    os.remove(os.path.join("saves/cam_calibration", f))

            np.savez(
                f"saves/cam_calibration/angle_{angle_deg}_disp_{displacement_cm}.npz",
                angle=angle_deg,
                displacement=displacement_cm,
            )
            print(f"Saved calibration: angle={angle_deg}, disp={displacement_cm}")
            break

    cv2.destroyAllWindows()

    for cam in cams:
        cam.close()



if __name__ == "__main__":
    main()
