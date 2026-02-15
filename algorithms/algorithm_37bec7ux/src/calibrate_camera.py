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
        cv2.resizeWindow(self.window_name, 1500, 1050)

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

        right_text = "W/S: Angle   A/D: Disp   Q: Quit   X: Save"
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

    cam_info = cams[0].get_camera_information()
    resolution = cam_info.camera_configuration.resolution
    w = min(720, resolution.width)
    h = min(404, resolution.height)
    low_res = sl.Resolution(w, h)

    calibration_params = cam_info.camera_configuration.calibration_parameters
    fx = calibration_params.left_cam.fx
    fy = calibration_params.left_cam.fy

    intr = ransac.Intrinsics(w / 2, h / 2, fx / 2, fy / 2)

    drive_conf = ransac.GridConfiguration(5000, 5000, 50, thres=2)
    block_conf = ransac.GridConfiguration(5000, 5000, 50, thres=1)

    image_mats = [sl.Mat(), sl.Mat()]
    depth_mats = [sl.Mat(), sl.Mat()]

    ui = CameraMergeUI(grid_size=60)
    angle_deg = 0
    displacement_cm = 0

    key = 0

    while key != 113:

        occ_grids = []

        for i in range(2):

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

            real_coeffs = ransac.plane.real_coeffs(px_coeffs, intr)

            drive_ppc = ransac.occu.create_point_cloud(ransac_output, depths)
            drive_rpc = ransac.occu.pixel_to_real(drive_ppc, real_coeffs, intr)

            block_ppc = ransac.occu.create_point_cloud(
                ransac_output != 1, depths
            )
            block_rpc = ransac.occu.pixel_to_real(
                block_ppc, real_coeffs, intr
            )

            drive_occ = ransac.occu.occupancy_grid(drive_rpc, drive_conf)
            block_occ = ransac.occu.occupancy_grid(block_rpc, block_conf)

            full_occ = ransac.occu.composite(drive_occ, block_occ)

            occ_grids.append(full_occ)

        if len(occ_grids) < 2:
            continue

        full_occ_left = occ_grids[0]
        full_occ_right = occ_grids[1]

        h_occ, w_occ = full_occ_left.shape

        transform_left = cv2.getRotationMatrix2D(
            (w_occ // 2, h_occ // 2),
            angle_deg / 2,
            1
        )
        transform_left[0, 2] -= displacement_cm / 3.3 / 2

        transform_right = cv2.getRotationMatrix2D(
            (w_occ // 2, h_occ // 2),
            -angle_deg / 2,
            1
        )
        transform_right[0, 2] += displacement_cm / 3.3 / 2

        occ1 = cv2.warpAffine(
            full_occ_left,
            transform_left,
            (w_occ, h_occ),
            flags=cv2.INTER_LINEAR
        )

        occ2 = cv2.warpAffine(
            full_occ_right,
            transform_right,
            (w_occ, h_occ),
            flags=cv2.INTER_LINEAR
        )

        merged_occ = np.maximum(occ1, occ2)
        merged_occ = np.where(
            (occ1 == 128) | (occ2 == 128),
            np.maximum(occ1, occ2),
            merged_occ
        )

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
                transform_left=transform_left,
                transform_right=transform_right
            )
            print(f"Saved calibration: angle={angle_deg}, disp={displacement_cm}")
            break

    cv2.destroyAllWindows()

    for cam in cams:
        cam.close()



if __name__ == "__main__":
    main()
