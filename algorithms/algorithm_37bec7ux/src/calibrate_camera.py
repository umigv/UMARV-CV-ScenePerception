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
        return cv2.applyColorMap(grid * 255, cv2.COLORMAP_TURBO)

    def _draw_point_cloud(self, pc, color):
        img = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.uint8)
        for x, y in pc:
            if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                cv2.circle(img, (x, y), 1, color, -1)
        return img

    def _make_panel(self, title, img, title_color):
        panel = np.ones((self.panel_size, self.panel_size, 3), dtype=np.uint8) * 42

        self._draw_centered_text(
            panel,
            title,
            y=32,
            scale=0.48,
            color=title_color
        )

        h, w = img.shape[:2]
        y0 = (self.panel_size - h) // 2 + 26
        x0 = (self.panel_size - w) // 2
        panel[y0:y0 + h, x0:x0 + w] = img
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

        right_text = "W/S: Angle   A/D: Disp   Q: Quit"
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
        pc1,
        pc2,
        merged_occ,
        merged_pc,
        angle,
        displacement,
    ):
        occ_row = np.hstack([
            self._make_panel("Camera 1 - Occ", self._colorize_grid(occ1), (255, 120, 120)),
            self._make_panel("Camera 2 - Occ", self._colorize_grid(occ2), (120, 255, 120)),
            self._make_panel("Merged - Occ", self._colorize_grid(merged_occ), (220, 220, 220)),
        ])

        pc_row = np.hstack([
            self._make_panel("Camera 1 - Points", self._draw_point_cloud(pc1, (255, 0, 0)), (255, 120, 120)),
            self._make_panel("Camera 2 - Points", self._draw_point_cloud(pc2, (0, 255, 0)), (120, 255, 120)),
            self._make_panel("Merged - Points", self._draw_point_cloud(merged_pc, (255, 255, 0)), (220, 220, 220)),
        ])

        width = occ_row.shape[1]

        title_bar = self._draw_title_bar(width)
        camera_view = self._draw_camera_top_view(width, angle, displacement)
        controls = self._draw_controls_bar(width, angle, displacement)

        dashboard = np.vstack([
            title_bar,
            occ_row,
            pc_row,
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
    

def main():
    GRID_SIZE = 160
    NUM_POINTS = 350

    ui = CameraMergeUI(grid_size=GRID_SIZE)

    angle = 20
    displacement = 5
    occ1 = np.random.randint(0, 2, (GRID_SIZE, GRID_SIZE), np.uint8)
    occ2 = np.random.randint(0, 2, (GRID_SIZE, GRID_SIZE), np.uint8)

    pc1 = np.random.randint(0, GRID_SIZE*2, (NUM_POINTS, 2))
    pc2 = np.random.randint(0, GRID_SIZE*2, (NUM_POINTS, 2))


    merged_occ = np.maximum(occ1, occ2)
    merged_pc = np.vstack([pc1, pc2])

    while True:
            

        ui.render(
            occ1=occ1,
            occ2=occ2,
            pc1=pc1,
            pc2=pc2,
            merged_occ=merged_occ,
            merged_pc=merged_pc,
            angle=angle,
            displacement=displacement,
        )

        key, angle, displacement = ui.handle_keyboard(angle, displacement)

        if key in (27, ord('q')):
            break

    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()
