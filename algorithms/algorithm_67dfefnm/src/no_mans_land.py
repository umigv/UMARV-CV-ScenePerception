import os

import cv2
import numpy as np

from hsv import hsv

class NoMansLand:
    def __init__(self, debug = False):
        self.hsv_obj = None
        self.image = None

        self.ramp_mask = None
        self.ramp_cnt = None

        self.final = None
        self.centroid = (None, None)

        self.debug = debug

    def update_mask(self):
        #defining the ranges for HSV values
        self.final, dict = self.hsv_obj.get_mask(self.image)
        
        self.ramp_mask = dict["ramp_color"]

    def is_ramp_visible(self):
        cnts, _ = cv2.findContours(self.ramp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_area = 1500
        max_found = 0

        best_cnt = None

        for cnt in cnts:
            area = cv2.contourArea(cnt)
            if (area > min_area) and (area > max_found):
                max_found = area
                best_cnt = cnt
        
        if best_cnt is not None:
            self.ramp_cnt = best_cnt
            return True
        else:  
            return False
            
    def magnitude_of_scalar_projection(self, of: tuple[float, float], onto: tuple[float, float]):
        onto_mag = ((onto[0] ** 2) + (onto[1] ** 2)) ** 0.5
        of_dot_onto = (of[0] * onto[0]) + (of[1] * onto[1])

        return abs(of_dot_onto / onto_mag)

    def grab_ramp_corners(self):
        diag_length = ((self.width ** 2) + (self.height ** 2)) ** 0.5

        # bottom left
        bl_ref_vec = (1.0, -1.5)
        most_bl_point = (self.ramp_cnt[0, 0, 0], self.ramp_cnt[0, 0, 1])
        min_bl_dist = diag_length

        # bottom right
        br_ref_vec = (-1.0, -1.5)
        most_br_point = (self.ramp_cnt[0, 0, 0], self.ramp_cnt[0, 0, 1])
        min_br_dist = diag_length

        # top left
        tl_ref_vec = (1.0, 2.0)
        most_tl_point = (self.ramp_cnt[0, 0, 0], self.ramp_cnt[0, 0, 1])
        min_tl_dist = diag_length

        # top right
        tr_ref_vec = (-1.0, 2.0)
        most_tr_point = (self.ramp_cnt[0, 0, 0], self.ramp_cnt[0, 0, 1])
        min_tr_dist = diag_length

        for point in self.ramp_cnt:
            x, y = float(point[0, 0]), float(point[0, 1])

            # bottom left
            bl_diff_vec = (x, -1 * (self.height - 1 - y))
            bl_dist = self.magnitude_of_scalar_projection(bl_diff_vec, bl_ref_vec)
            if bl_dist < min_bl_dist:
                min_bl_dist = bl_dist
                most_bl_point = (int(x), int(y))

            # bottom right
            br_diff_vec = (-1 * (self.width - 1 - x), -1 * (self.height - 1 - y))
            br_dist = self.magnitude_of_scalar_projection(br_diff_vec, br_ref_vec)
            if br_dist < min_br_dist:
                min_br_dist = br_dist
                most_br_point =(int(x), int(y))

            # top left
            tl_diff_vec = (x, y)
            tl_dist = self.magnitude_of_scalar_projection(tl_diff_vec, tl_ref_vec)
            if tl_dist < min_tl_dist:
                min_tl_dist = tl_dist
                most_tl_point = (int(x), int(y))

            # top right
            tr_diff_vec = (-1 * (self.width - 1 - x), y)
            tr_dist = self.magnitude_of_scalar_projection(tr_diff_vec, tr_ref_vec)
            if tr_dist < min_tr_dist:
                min_tr_dist = tr_dist
                most_tr_point = (int(x), int(y))

        return most_bl_point, most_br_point, most_tl_point, most_tr_point
            
    
    def state_1(self): # can't see ramp
        self.centroid = (self.width // 2, 40)

    def state_2(self): # seeing ramp
        bottom_left, bottom_right, top_left, top_right = self.grab_ramp_corners()

        mid_bottom = ((bottom_left[0] + bottom_right[0]) // 2, (bottom_left[1] + bottom_right[1]) // 2)
        mid_top = ((top_left[0] + top_right[0]) // 2, (top_left[1] + top_right[1]) // 2)
        mid = ((mid_bottom[0] + mid_top[0]) // 2, (mid_bottom[1] + mid_top[1]) // 2)

        if self.ramp_mask[self.height - 15, self.width // 2] != 0: # we are probably on the ramp now
            self.centroid = (mid_top[0], mid_top[1] - 50)
            cv2.line(self.final, top_left, (0, 0), 255, 10)
            cv2.line(self.final, top_right, (self.width, 0), 255, 10)
            cv2.circle(self.final, mid, 10, 128, -1)
        else:
            self.centroid = (mid_bottom[0], min(self.height - 1, mid_bottom[1] + 50))
            cv2.line(self.final, bottom_left, (0, self.height), 255, 10)
            cv2.line(self.final, bottom_right, (self.width, self.height), 255, 10)
            cv2.circle(self.final, mid, 10, 128, -1)

        if self.debug:
            cv2.circle(self.final, bottom_left, 7, 128, -1)
            cv2.circle(self.final, bottom_right, 7, 128, -1)
            cv2.circle(self.final, top_left, 7, 128, -1)
            cv2.circle(self.final, top_right, 7, 128, -1)

            cv2.circle(self.final, mid_bottom, 10, 128, -1)
            cv2.circle(self.final, mid_top, 10, 128, -1)
            cv2.circle(self.final, mid, 10, 128, -1)

    def state_machine(self):
        ramp_visible = False
        ramp_visible = self.is_ramp_visible()

        if ramp_visible:
            if self.debug:
                print("ramp mode")
            self.state_2()
        else:
            self.state_1()

    def run(self):
        '''
        script_dir = os.path.dirname(os.path.abspath(__file__))

        vid = "output.mp4"
        path = os.path.join(script_dir, "data", vid)  # Always resolves correctly
        cap = cv2.VideoCapture(path)
        # vid = "output.mp4"
        # path = "data/" + vid
        # cap = cv2.VideoCapture(path) # 0 for webcam, 1,2 for external cameras
        self.hsv_obj = hsv("output.mp4")
        '''
        script_dir = os.path.dirname(os.path.abspath(__file__))
        vid = "output.mp4"
        local_path = os.path.join(script_dir, "data", vid)
        root_path = os.path.join(script_dir, "..", "..", "..", "data", vid)
        path = local_path if os.path.exists(local_path) else root_path

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video at:\n  {local_path}\n  {root_path}")

        self.hsv_obj = hsv(path)  # ← fixed
        self.hsv_obj.tune("ramp_color")

        # green reference
        # "ramp_color": {
        #     "h_upper": 107,
        #     "h_lower": 61,
        #     "s_upper": 151,
        #     "s_lower": 61,
        #     "v_upper": 231,
        #     "v_lower": 120
        # }

        # you need a JSON entry with the same name as your video to use this
        # you can rename the "green-reference" entry to your video temporarily to tune it
                    
        while cap.isOpened():
            try: 
                ret, self.image = cap.read()
                self.height, self.width, _ = self.image.shape
            except AttributeError:
                print("Program finished running")

            if ret:
                self.update_mask()
                self.state_machine()

                cv2.circle(self.final, self.centroid, 5, 255, -1)

                cv2.namedWindow("Video", cv2.WINDOW_NORMAL)
                cv2.imshow("Video", self.image)
                cv2.namedWindow("mask", cv2.WINDOW_NORMAL)
                cv2.imshow("mask", self.final)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break

        cap.release()
        cv2.destroyAllWindows()
            
    def run_frame(self, frame, hsv_indentifier):
        if self.hsv_obj is None:
            self.hsv_obj = hsv(hsv_indentifier)
            
        self.image = frame
        self.height, self.width, _ = self.image.shape
        
        self.update_mask()
        self.state_machine()

        cv2.circle(self.final, self.centroid, 5, 255, -1)

        cv2.namedWindow("Video", cv2.WINDOW_NORMAL)
        cv2.imshow("Video", self.image)
        cv2.namedWindow("mask", cv2.WINDOW_NORMAL)
        cv2.imshow("mask", self.final)

def main():
    obj = NoMansLand(debug = True)
    obj.run()

if __name__ == "__main__":
    main()
