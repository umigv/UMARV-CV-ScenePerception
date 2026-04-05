import cv2
import numpy as np
from hsv import hsv


class CurvedLanekeeping:
    # Left and right bounds should be kept symmetric
    def __init__(self, debug: bool = False, barrels: bool = True,
                 barrel_mode: str = "YOLO",
                 left_bounds: tuple[float, float] = (0.15, 0.45),
                 right_bounds: tuple[float, float] = (0.55, 0.85),
                 vertical_bounds: tuple[float, float] = (0.2, 0.8)):
        self.image = None
        self.hsv_image = None

        self.white_mask = None
        self.yellow_mask = None

        self.final = None

        self.hsv_obj = None

        self.centroid = (None, None)

        self.width = None
        self.height = None

        self.look_for_barrels = barrels
        self.barrel_mode = barrel_mode

        self.left_bounds = left_bounds
        self.right_bounds = right_bounds
        self.vertical_bounds = vertical_bounds

        self.debug = debug

    def update_mask(self):
        #defining the ranges for HSV values
        self.final, dict = self.hsv_obj.get_mask(self.image, yolo_barrels=self.look_for_barrels)

        # print(dict)
        
        self.white_mask = dict["white"]
        self.yellow_mask = dict["yellow"]
        if self.barrel_mode != "YOLO":
          self.barrel_color_mask = dict["orange"]

        # final_bgr = cv2.cvtColor(self.final, cv2.COLOR_GRAY2BGR)
        # combined = np.hstack((self.image, final_bgr))
        # cv2.namedWindow("Combined Image", cv2.WINDOW_NORMAL)
        # cv2.imshow("Combined Image", self.final)

    def state_machine(self):
        # looking for barrel
        if self.hsv_obj.barrel_boxes is not None:
            for segment in self.hsv_obj.barrel_boxes:
                x_min, y_min, x_max, y_max = segment
                vertices = np.array([
                    [x_min * self.width, y_min * self.height],
                    [x_max * self.width, y_min * self.height],
                    [x_max * self.width, y_max * self.height],
                    [x_min * self.width, y_max * self.height]
                ], dtype=np.int32)

                if self.debug:
                    # print(vertices)
                    cv2.rectangle(self.final, vertices[0], vertices[2], 127, 5)
                
                if(y_min * self.height > self.height // 2):
                    midpoint = (x_max * self.width) - (x_min * self.width)
                    if(midpoint > self.width // 4 and midpoint < (self.width - (self.width // 4))):
                        self.centroid = midpoint
                        return

        # normal state
        left_min = int(self.left_bounds[0] * self.width)
        left_max = int(self.left_bounds[1] * self.width)

        right_min = int(self.right_bounds[0] * self.width)
        right_max = int(self.right_bounds[1] * self.width)

        vert_min = int(self.vertical_bounds[0] * self.height)
        vert_max = int(self.vertical_bounds[1] * self.height)

        best_left_point = None
        min_left_y = vert_max
        best_right_point = None
        min_right_y = vert_max

        cnts, _ = cv2.findContours(self.white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in cnts:
            for point in contour:
                x = point[0, 0]
                y = point[0, 1]

                valid_left_point = x < left_max and x > left_min \
                  and y > vert_min and y < vert_max
                
                valid_right_point = x < right_max and x > right_min \
                  and y > vert_min and y < vert_max

                if valid_left_point:
                    if y < min_left_y:
                        min_left_y = y
                        best_left_point = (x, y)

                if valid_right_point:
                    if y < min_right_y:
                        min_right_y = y
                        best_right_point = (x, y)
      
        if best_left_point is not None and best_right_point is not None:    
            self.centroid = (
                (best_left_point[0] + best_right_point[0]) // 2,
                (best_left_point[1] + best_right_point[1]) // 2
            )
        else: # fallback
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

        vert_min = int(self.vertical_bounds[0] * self.height)
        vert_max = int(self.vertical_bounds[1] * self.height)

        cv2.line(self.final, (left_min, vert_min),
            (left_min, vert_max),
            color, 10)
        cv2.line(self.final, (left_max, vert_min),
            (left_max, vert_max),
            color, 10)
        cv2.line(self.final, (left_min, vert_min), 
            (left_max, vert_min), color, 10)
        cv2.line(self.final, (left_min, vert_max), 
            (left_max, vert_max), color, 10)
      
        cv2.line(self.final, (right_min, vert_min),
            (right_min, vert_max),
            color, 10)
        cv2.line(self.final, (right_max, vert_min),
            (right_max, vert_max),
            color, 10)
        cv2.line(self.final, (right_min, vert_min), 
            (right_max, vert_min), color, 10)
        cv2.line(self.final, (right_min, vert_max), 
            (right_max, vert_max), color, 10)

    def run(self):
        cap = cv2.VideoCapture("data/left_curved_road.MOV")
        self.hsv_obj = hsv("data/left_curved_road.MOV", barrel_mode = self.barrel_mode)

        # self.hsv_obj.tune("white")
        # self.hsv_obj.tune("yellow")
        # self.hsv_obj.tune("orange")
        
        while cap.isOpened():
            ret, self.image = cap.read()
            if ret:
                self.height, self.width, _ = self.image.shape
                
                self.update_mask()
                self.state_machine()

                if self.debug:
                    self.show_search_boxes(150)
                    
                cv2.circle(self.final, self.centroid, 5, 255, -1)

                cv2.namedWindow("Final Mask", cv2.WINDOW_NORMAL)
                cv2.imshow("Final Mask", self.final)
                cv2.namedWindow("Yellow Mask", cv2.WINDOW_NORMAL)
                cv2.imshow("Yellow Mask", self.yellow_mask)
                cv2.namedWindow("White Mask", cv2.WINDOW_NORMAL)
                cv2.imshow("White Mask", self.white_mask)
                cv2.namedWindow("White Mask", cv2.WINDOW_NORMAL)
                cv2.imshow("White Mask", self.white_mask)
                if self.barrel_mode != "YOLO":
                    cv2.namedWindow(f"{self.barrel_mode} Mask", cv2.WINDOW_NORMAL)
                    cv2.imshow(f"{self.barrel_mode} Mask", self.barrel_color_mask)


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

        cv2.circle(self.final, self.centroid, 5, 255, -1)
        cv2.imshow("Final Mask", self.final)

        return self.final, self.centroid

def main():
    obj = CurvedLanekeeping(debug = False, barrel_mode = "orange")
    obj.run()

if __name__ == "__main__":
    main()