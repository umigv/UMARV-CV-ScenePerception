import cv2
import pytesseract
from ultralytics import YOLO
import numpy as np


model = YOLO('./data/stopsigns.pt')


custom_oem_psm_config = r'--oem 3 --psm 6'

# seek out the words from comp **ADD THE NAME OF FAKES!
with open('user_words.txt', 'w') as f:
    f.write("IGVC\nSOUP\nSTOP")

# pick video stream
cap = cv2.VideoCapture("./data/stopvid.mp4")  
frame_count = 0
process_per_frame = 10
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    if frame_count % process_per_frame != 0:
        continue
    results = model(frame)
    
    ###flag variable starts as false, no stop sign seen yet
    stop_found = False

    for result in results:
        boxes = result.boxes.xyxy.tolist()
        confidences = result.boxes.conf.tolist()
        class_ids = result.boxes.cls.tolist()
        
        for box, confidence, class_id in zip(boxes, confidences, class_ids):
            if confidence > 0.8:  # confidence at least 80% (can mess with, dont want too low or picks up a  lot of noise)
                x1, y1, x2, y2 = map(int, box)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                #make bounds
                
                cropped = frame[y1:y2, x1:x2]
                
                # red mask
                hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
                lower_red1 = np.array([0, 70, 50])
                upper_red1 = np.array([10, 255, 255])
                lower_red2 = np.array([160, 70, 50])
                upper_red2 = np.array([180, 255, 255])
                mask = cv2.bitwise_or(
                    cv2.inRange(hsv, lower_red1, upper_red1),
                    cv2.inRange(hsv, lower_red2, upper_red2)
                )
                #find red 
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    red_cropped = cropped[y:y+h, x:x+w]
                    
                    #ratio-based crop, ESTIMATION of best based on trial error
                    if red_cropped.size > 0:
                        h, w = red_cropped.shape[:2]
                        final_crop = red_cropped[int(h/4):int(h-h/4), int(w/27):int(w-w/27)]
                        
                        #OCR
                        gray = cv2.cvtColor(final_crop, cv2.COLOR_BGR2GRAY)
                        inverted = cv2.bitwise_not(gray)
                        _, thresholded = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                        # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

                        text = pytesseract.image_to_string(thresholded, config=custom_oem_psm_config).strip()
                        

                        ######flag variable set if it is in fact a stop sign
                        if text.lower() == "stop":
                            stop_found = True
                        else:
                            print("not stop sign")
                            print(text.lower())



                        #display the text found
                        cv2.putText(frame, f"{text} ({confidence:.2f})", (x1, y1+10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        if stop_found:
                            cv2.putText(frame, f"Stop Sign", (x1, y1+40), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        else:
                            cv2.putText(frame, f"Unknown {text.lower()}", (x1, y1+40), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        print(stop_found)
                        print(confidence)
            else:
                print("Unknown")
                cv2.putText(frame, f"Unknown", (100, 100), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
    #display stuff for GUI/ not necessary for functionality
    cv2.imshow('Stop Sign Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()