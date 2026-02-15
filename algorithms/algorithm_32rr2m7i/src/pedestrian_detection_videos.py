import cv2
import os
from ultralytics import YOLO
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# default YOLO model already has person as a class
model = YOLO('./data/yolov8n.pt')

# init directories
cap = cv2.VideoCapture("./data/pedestrian_1.mp4")  
frame_count = 0
process_per_frame = 3
while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break
    frame_count += 1
    if frame_count % process_per_frame != 0:
        continue

    # run image through model
    results = model(img)

    for results in results:
        boxes = results.boxes.xyxy.tolist()
        confidences = results.boxes.conf.tolist()
        class_ids = results.boxes.cls.tolist()

        for box, confidence, class_id in zip(boxes, confidences, class_ids):
            # class 0 is person (built in) and adjust confidence as needed
            if class_id == 0 and confidence > 0.7:
                px1, py1, px2, py2 = map(int, box)

                # crop to person only from yolo output
                # cropped_image = img[py1:py2, px1:px2]
                cropped_image = img[py1:int((py1+(py2-py1)/2)), px1:px2]

                # analyze colors
                img_HSV = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2HSV)
                
                # apply orange mask
                lower_orange = np.array([0, 120, 100])
                upper_orange = np.array([12, 255, 255])

                mask = cv2.inRange(img_HSV, lower_orange, upper_orange)

                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                if contours:
                    # sees largest orange area which will be the vest
                    # largest_countour = max(contours, key = cv2.contourArea)
                    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
                    # if we have 1 contour for the vest
                    if len(contours) == 1:
                        print(sorted_contours[0])
                        largest_coutour = sorted_contours[0]
                        print(sorted_contours[0])
                        second_largest_contour = sorted_contours[0]
                    # if we have 2 contours (vest is split in 2)
                    elif len(contours) > 1:
                        print(sorted_contours[0])
                        largest_coutour = sorted_contours[0]
                        print(sorted_contours[1])
                        second_largest_contour = sorted_contours[1]
                
                    

                    vest_x1, vest_y1, vest_w1, vest_h1 = cv2.boundingRect(largest_coutour)
                    vest_x2, vest_y2, vest_w2, vest_h2 = cv2.boundingRect(second_largest_contour)

                    min_x = min(vest_x1,vest_x2)
                    min_y = min(vest_y1,vest_y2)
                    max_x = max(vest_x1 + vest_w1, vest_x2 + vest_w2)
                    max_y = max(vest_y1 + vest_h1, vest_y2 + vest_h2)

                    merged_vest_x1 = px1 + min_x
                    merged_vest_y1 = py1 + min_y
                    merged_vest_x2 = px1 + max_x
                    merged_vest_y2 = py1 + max_y

                    cv2.rectangle(img, (merged_vest_x1, merged_vest_y1), (merged_vest_x2, merged_vest_y2), (0, 165, 255), 2)
                    cv2.putText(img, f"orange vest", (merged_vest_x1, merged_vest_y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                    
                    cv2.rectangle(img, (px1, py1), (px2, py2), (0, 255, 0), 2)
                    cv2.putText(img, f"orange vest person", (px1, py1-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                
    cv2.imshow('Safety Vest Detection', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


    
# applied_mask = cv2.bitwise_and(img, img, mask = mask)
# cv2.imshow(img_HSV)
# cv2.imshow(applied_mask)
