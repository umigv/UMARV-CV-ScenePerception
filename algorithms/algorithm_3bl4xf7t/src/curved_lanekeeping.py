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

        self.look_for_barrels = False

        # These should be kept symmetric
        self.left_bounds = (0.15, 0.45)
        self.right_bounds = (0.55, 0.85)
        
        self.vertical_min = 0.2
        self.vertical_max = 0.8

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
        best_left_point = None
        min_left_y = self.vertical_max * self.height
        best_right_point = None
        min_right_y = self.vertical_max * self.height

        cnts, _ = cv2.findContours(self.white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in cnts:
            for point in contour:
                valid_left_point = point[0, 0] < self.left_bounds[1] * self.width and point[0, 0] > self.left_bounds[0] * self.width \
                  and point[0, 1] > self.vertical_min * self.height and point[0, 1] < self.vertical_max * self.height
                
                valid_right_point = point[0, 0] < self.right_bounds[1] * self.width and point[0, 0] > self.right_bounds[0] * self.width \
                  and point[0, 1] > self.vertical_min * self.height and point[0, 1] < self.vertical_max * self.height

                if valid_left_point:
                    if point[0, 1] < min_left_y:
                        min_left_y = point[0, 1]
                        best_left_point = (point[0, 0], point[0, 1])

                if valid_right_point:
                    if point[0, 1] < min_right_y:
                        min_right_y = point[0, 1]
                        best_right_point = (point[0, 0], point[0, 1])
      
        if best_left_point is not None and best_right_point is not None:    
            self.centroid = (
                (best_left_point[0] + best_right_point[0]) // 2,
                (best_left_point[1] + best_right_point[1]) // 2
            )
        else:
            self.centroid = (
                self.width // 2,
                self.height // 2
            )

        if self.debug:
            cv2.circle(self.final, best_left_point, 10, 128, -1)
            cv2.circle(self.final, best_right_point, 10, 128, -1)

    def show_search_boxes(self, color: int) -> None:
        left_min = int(self.left_bounds[0] * self.width)
        left_max = int(self.left_bounds[1] * self.width)

        right_min = int(self.right_bounds[0] * self.width)
        right_max = int(self.right_bounds[1] * self.width)

        y_min = int(self.vertical_min * self.height)
        y_max = int(self.vertical_max * self.height)

        cv2.line(self.final, (left_min, y_min),
            (left_min, y_max),
            128, 10)
        cv2.line(self.final, (left_max, y_min),
            (left_max, y_max),
            128, 10)
        cv2.line(self.final, (left_min, y_min), 
            (left_max, y_min), 128, 10)
        cv2.line(self.final, (left_min, y_max), 
            (left_max, y_max), 128, 10)
      
        cv2.line(self.final, (right_min, y_min),
            (right_min, y_max),
            128, 10)
        cv2.line(self.final, (right_max, y_min),
            (right_max, y_max),
            128, 10)
        cv2.line(self.final, (right_min, y_min), 
            (right_max, y_min), 128, 10)
        cv2.line(self.final, (right_min, y_max), 
            (right_max, y_max), 128, 10)

    def run(self):
        cap = cv2.VideoCapture("data/left_curved_road.MOV")
        self.hsv_obj = hsv("data/left_curved_road.MOV")

        # self.hsv_obj.tune("white")
        # self.hsv_obj.tune("yellow")
        
        while cap.isOpened():
            ret, self.image = cap.read()
            if ret:
                self.height, self.width, _ = self.image.shape
                
                self.update_mask()
                self.state_machine()

                if self.debug:
                    self.show_search_boxes(150)
                    
                cv2.circle(self.final, self.centroid, 5, 100, -1)

                cv2.namedWindow("Final Mask", cv2.WINDOW_NORMAL)
                cv2.imshow("Final Mask", self.final)
                cv2.namedWindow("Yellow Mask", cv2.WINDOW_NORMAL)
                cv2.imshow("Yellow Mask", self.yellow_mask)
                cv2.namedWindow("White Mask", cv2.WINDOW_NORMAL)
                cv2.imshow("White Mask", self.white_mask)

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
        self.state_machine()

        cv2.circle(self.final, self.centroid, 5, 100, -1)
        cv2.imshow("Final Mask", self.final)

        return self.final, self.centroid

def main():
    obj = CurvedLanekeeping(debug = True)
    obj.run()

if __name__ == "__main__":
    main()