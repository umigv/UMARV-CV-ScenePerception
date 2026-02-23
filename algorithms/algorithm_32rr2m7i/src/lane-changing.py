import numpy as np
import cv2
from ultralytics import YOLO

class SolidStateMachine:
    def __init__(self):
        # Keep original variable names exactly as provided
        self.person_model = YOLO('./data/yolov8n.pt')
        self.lines_model = YOLO('./data/best_yolov11_lane_lines.pt')
        # self.lines_model = YOLO('./data/combinedv2.pt')
        self.cap = cv2.VideoCapture("data/9 function test pedestrian detection lane change & barrel stop.MP4") 
        self.frame_count = 0
        self.process_per_frame = 3

        # Values for HSV
        self.white_lower_bound = np.array([0, 0, 180])
        self.white_upper_bound = np.array([179, 103, 255])

        # State constants
        self.state_1 = 1
        self.state_2 = 2
        self.state_3 = 3
        self.state = self.state_1

    def change_lanes(self, capture, img, y_waypoint):
        # Placeholder logic
        # blank_image = np.zeros(img.shape[:2], dtype=np.uint8)
        # results = self.lines_model(img)
        # r = results[0]
        # for mask, class_id in zip(r.masks.data, r.boxes.cls):
        #     if int(class_id) == 0:
        #         pixels_of_class = mask.cpu().numpy()
        #         blank_image[pixels_of_class  > 0.5] = 255

        # new_img = results[0].plot()
        # hsv_image = cv2.cvtColor(new_img, cv2.COLOR_BGR2HSV)
            
        # # update masks
        # white_mask = cv2.inRange(hsv_image, self.white_lower_bound, self.white_upper_bound)
        
        # white_mask = cv2.resize(white_mask, (img.shape[1], img.shape[0]))
        
        # white_mask = cv2.erode(white_mask, None, iterations=1)
        # white_mask = cv2.dilate(white_mask, None, iterations=2)
        
        # mask = white_mask
        # height, width = mask.shape[:2]
        # mask[0:height//2, :] = 0 # top half of img is set to black        

        # cv2.imshow("masked", blank_image)

        results = self.lines_model(img)
    
        # Create an empty black mask the same size as the image
        full_mask = np.zeros(img.shape[:2], dtype=np.uint8)
        
        result = results[0]
        if result.masks is not None:
            # Loop through each detected object
            for mask, cls in zip(result.masks.data, result.boxes.cls):
                if int(cls) == 0:
                    # result.masks.data is usually lower resolution, 
                    # we convert to numpy and resize to match original image
                    m = mask.cpu().numpy()
                    m = cv2.resize(m, (img.shape[1], img.shape[0]))
                    
                    # Add this object's pixels to our full mask
                    full_mask[m > 0.5] = 255        
        
        x = self.find_x(y_waypoint, full_mask)
        
        
        return False, x, full_mask

    def sees_pedestrian_in_lane(self, capture, img):
        
        results = self.person_model(img)
        
        py2 = 0

        for results in results:
            boxes = results.boxes.xyxy.tolist()
            confidences = results.boxes.conf.tolist()
            class_ids = results.boxes.cls.tolist()

            for box, confidence, class_id in zip(boxes, confidences, class_ids):
                # class 0 is person (built in) and adjust confidence as needed
                px1, py1, px2, py2 = map(int, box)
                # self.change_lanes(capture, img, py2)
                if class_id == 0 and confidence > 0.7:
                    
                    print("PERSON")
                    height, width = img.shape[:2]
                    size_person = (px2-px1)/width
                    
                    
                    cv2.rectangle(img, (px1, py1), (px2, py2), (0, 255, 0), 2)
                    cv2.imshow("label", img)
                    if size_person > 0.12:
                        print("person within range")
                        return True, py2
                    else:
                        return False, py2
                    

                
        return False, py2

    def at_barrel(self, capture, img):
        # Placeholder logic
        return False
    
    def add_waypoint(self, y, img, x):
        x= 100
        center = (x, y)
        radius = 100
        color = [255,100,0]
        cv2.circle(img, center, radius, color, thickness=1, lineType=8, shift=0)
        # cv2.imshow("waypoint",img)
    def find_x(self, py2, img):
        # x = 100
        x_left = 100
        x_right = 100
        # for y in range(py2 - 20, py2 + 20):
        row = img[py2, :]
        height, width = img.shape
        # finding from left
        for i in range(width):
            cv2.circle(img, (i, py2), 3, 255, thickness=1)
            cv2.imshow("in_find_x", img)
            cv2.waitKey(1)
            # print(f"Row[i] : {row[i]}")
            if (row[i] == 255):
                x_left = i
                print("255-")
                break
        for i in range(width):
            print("in loop from right")
            cv2.circle(img, (i, py2), 3, 255, thickness=1)
            # print(f"Row[i] : {row[width - i - 1]}")
            # b, g,r = row[width -i - 1]
            # if b == 255:
            #     x_right = width - i - 1
            #     break
            if (row[width -i - 1] == 255):
                x_right = width - i - 1
                break


        x = int((x_left + x_right)/2) 

            # row = [1, 2, 3, 4]
            # for el in row[::-1]:
            #     print(el)
            # # 4, 3, 2, 1

        return x

    def run(self):
        running = True
        while(running and self.cap.isOpened()):
            # read frames
            ret, img = self.cap.read()
            if not ret:
                break
            
            self.frame_count += 1
            if self.frame_count % self.process_per_frame != 0:
                continue

            # State Logic
            if(self.state == self.state_1):
                see_pedestrian, y_waypoint = self.sees_pedestrian_in_lane(self.cap, img)
                
                if(see_pedestrian):
                    print("PERSON DETECTED")
                    self.state = self.state_2
                
            elif(self.state == self.state_2):
                done, x_waypoint, full_mask = self.change_lanes(self.cap, img, y_waypoint)
                self.add_waypoint(y_waypoint,full_mask, x_waypoint)
                # cv2.imshow("full mask", full_mask)
                if(done):
                    self.state = 3
                    
            elif(self.state == self.state_3):
                if(self.at_barrel(self.cap, img)):
                    running = False
                    print("AT BARREL")     
            
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.cap.release()
                cv2.destroyAllWindows()
                break

machine = SolidStateMachine()
machine.run()