import cv2
import numpy as np

from hsv import hsv

class NoMansLand:
    def __init__(self):
        self.hsv_obj = None
        self.image = None

        self.ramp_mask = None

        self.final = None
        self.centroid = (None, None)

    def update_mask(self):
        #defining the ranges for HSV values
        self.final, dict = self.hsv_obj.get_mask(self.image)
        
        self.ramp_mask = dict["ramp_color"]
        
        # self.past_stop_line()
        
        # if(self.done == False):
        # self.last_diff_y = self.find_left_most_lane()
        # else:
        #     self.draw_trapazoid()
        #     self.centroid = (self.width//2, 40)
        # self.find_center_of_lane()
        # cv2.namedWindow("mask", self.final)
        # cv2.imshow("mask", self.final)
        final_bgr = cv2.cvtColor(self.final, cv2.COLOR_GRAY2BGR)
        combined = np.vstack((self.image, final_bgr))
        # cv2.imshow("mask", combined)

    def is_ramp_visible(self):
        cnts, _ = cv2.findContours(self.ramp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_area = 1000
        for cnt in cnts:
            if cv2.contourArea(cnt) > min_area:
                return True, cnt
            
        return False, []
            
    def magntitude_of_scalar_projection(self, of: tuple[float, float], onto: tuple[float, float]):
        onto_mag = ((onto[0] ** 2) + (onto[1] ** 2)) ** 0.5
        of_dot_onto = of[0] * onto[0] + of[1] * onto[1]

        return abs(of_dot_onto / onto_mag)

    def grab_ramp_corners(self, best_cnt):
        diag_length = ((self.width ** 2) + (self.height ** 2)) ** 0.5

        bl_ref_vec = (1.0, -1.0)
        most_bl_point = (best_cnt[0, 0, 0], best_cnt[0, 0, 1])
        min_bl_dist = diag_length

        br_ref_vec = (-1.0, -1.0)
        most_br_point = (best_cnt[0, 0, 0], best_cnt[0, 0, 1])
        min_br_dist = diag_length

        tl_ref_vec = (1.0, 2.0)
        most_tl_point = (best_cnt[0, 0, 0], best_cnt[0, 0, 1])
        min_tl_dist = diag_length

        tr_ref_vec = (-1.0, 2.0)
        most_tr_point = (best_cnt[0, 0, 0], best_cnt[0, 0, 1])
        min_tr_dist = diag_length

        for point in best_cnt:
            x, y = float(point[0, 0]), float(point[0, 1])

            bl_diff_vec = (x, -1 * (self.height - 1 - y))
            bl_dist = self.magntitude_of_scalar_projection(bl_diff_vec, bl_ref_vec)
            if bl_dist < min_bl_dist:
                min_bl_dist = bl_dist
                most_bl_point = (int(x), int(y))

            br_diff_vec = (-1 * (self.width - 1 - x), -1 * (self.height - 1 - y))
            br_dist = self.magntitude_of_scalar_projection(br_diff_vec, br_ref_vec)
            if br_dist < min_br_dist:
                min_br_dist = br_dist
                most_br_point = (int(x), int(y))

            tl_diff_vec = (x, y)
            tl_dist = self.magntitude_of_scalar_projection(tl_diff_vec, tl_ref_vec)
            if tl_dist < min_tl_dist:
                min_tl_dist = tl_dist
                most_tl_point = (int(x), int(y))

            tr_diff_vec = (-1 * (self.width - 1 - x), y)
            tr_dist = self.magntitude_of_scalar_projection(tr_diff_vec, tr_ref_vec)
            if tr_dist < min_tr_dist:
                min_tr_dist = tr_dist
                most_tr_point = (int(x), int(y))

        return most_bl_point, most_br_point, most_tl_point, most_tr_point
            
    
    def state_1(self): # can't see ramp
        self.centroid = (self.width // 2, 40)

    def state_2(self, ramp_cnt): # on ramp
        bottom_left, bottom_right, top_left, top_right = self.grab_ramp_corners(ramp_cnt)

        cv2.line(self.final, bottom_left, (0, self.height), 255, 10)
        cv2.line(self.final, bottom_right, (self.width, self.height), 255, 10)

        mid_bottom = ((bottom_left[0] + bottom_right[0]) // 2, (bottom_left[1] + bottom_right[1]) // 2)
        mid_top = ((top_left[0] + top_right[0]) // 2, (top_left[1] + top_right[1]) // 2)
        cv2.circle(self.final, mid_bottom, 10, 128, -1)
        cv2.circle(self.final, mid_top, 10, 128, -1)

        cv2.circle(self.final, bottom_left, 10, 128, -1)
        cv2.circle(self.final, bottom_right, 10, 128, -1)
        cv2.circle(self.final, top_left, 10, 128, -1)
        cv2.circle(self.final, top_right, 10, 128, -1)

        print(bottom_left)
        print(bottom_right)
        print(top_left)
        print(top_right)
        print()
        
    def run(self):
        cap = cv2.VideoCapture('data/ramp.MOV') # 0 for webcam # 1,2 for external cameras
        self.hsv_obj = hsv("data/ramp.MOV")

        # self.hsv_obj.tune("ramp_color")
                    
        while cap.isOpened():
            ret, self.image = cap.read()
            self.height, self.width, _ = self.image.shape

            if ret:
                self.update_mask()

                ramp_visible = False
                ramp_visible, best_cnt = self.is_ramp_visible()

                if ramp_visible:
                    print("ramp mode")
                    self.state_2(best_cnt)
                else:
                    self.state_1()

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
            
    def run_frame(self, hsv_indentifier, frame):
        if self.hsv_obj is None:
            self.hsv_obj = hsv(hsv_indentifier)
            
        self.image = frame
        self.height, self.width, _ = self.image.shape
        
        self.update_mask()
    
    def detect_ramp(color_range):
        pass



def main():
    obj = NoMansLand()
    obj.run()
    # obj.detect_ramp(123) #fill with color range (green)

if __name__ == "__main__":
    main()