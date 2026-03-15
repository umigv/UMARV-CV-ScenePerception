import cv2
import numpy as np
from hsv import hsv


class CurvedLanekeeping:
    def __init__(self, debug = False):
        self.image = None
        self.hsv_image = None

        self.white_mask = None
        self.yellow_mask = None

        self.final = None

        self.hsv_obj = None

        self.centroid = (None, None)

        self.width = None
        self.height = None

        self.debug = debug

    def update_mask(self):
        #defining the ranges for HSV values
        self.final, dict = self.hsv_obj.get_mask(self.image, yolo_barrels=self.look_for_barrels and (not self.debug))

        # print(dict)
        
        self.white_mask = dict["white"]
        self.yellow_mask = dict["yellow"]

        # final_bgr = cv2.cvtColor(self.final, cv2.COLOR_GRAY2BGR)
        # combined = np.hstack((self.image, final_bgr))
        # cv2.namedWindow("Combined Image", cv2.WINDOW_NORMAL)
        # cv2.imshow("Combined Image", self.final)

    def state_machine(self):
        pass

    def run(self):
        cap = cv2.VideoCapture("data/right_turn_cropped.mp4")
        self.hsv_obj = hsv("data/right_turn_cropped.mp4")

        # "white": {
        #     "h_upper": 179,
        #     "h_lower": 0,
        #     "s_upper": 218,
        #     "s_lower": 0,
        #     "v_upper": 255,
        #     "v_lower": 212
        # },
        # "yellow": {
        #     "h_upper": 179,
        #     "h_lower": 23,
        #     "s_upper": 255,
        #     "s_lower": 150,
        #     "v_upper": 255,
        #     "v_lower": 200
        # }
        # backup of values from json

        # self.hsv_obj.tune("white")
        # self.hsv_obj.tune("yellow")
        
        while cap.isOpened():
            ret, self.image = cap.read()
            if ret:
                self.height, self.width, _ = self.image.shape
                
                self.update_mask()
                self.state_machine()

                cv2.circle(self.final, self.centroid, 5, 255, -1)

                cv2.namedWindow("Final Mask", cv2.WINDOW_NORMAL)
                cv2.imshow("Final Mask", self.final)
                cv2.namedWindow("Yellow Mask", cv2.WINDOW_NORMAL)
                cv2.imshow("Yellow Mask", self.yellow_mask)
                cv2.namedWindow("White Mask", cv2.WINDOW_NORMAL)
                cv2.imshow("White Mask", self.white_mask)

                if self.debug:
                    print()
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
        cap.release()
        cv2.destroyAllWindows()
        
    # >>> change: run_frame now runs full pipeline (HSV + state machine) and returns results
    def run_frame(self, hsv_indentifier, frame):
        if self.hsv_obj is None:
            self.hsv_obj = hsv(hsv_indentifier)
        
        self.image = frame
        self.height, self.width, _ = self.image.shape
    
        self.update_mask()
        self.state_machine()

        cv2.circle(self.final, self.centroid, 5, 255, -1)
        cv2.imshow("Final Mask", self.final)

        return self.final, self.centroid
    # <<< end of change

def main():
    obj = CurvedLanekeeping(debug = False)
    obj.run()

if __name__ == "__main__":
    main()