import cv2
import torch
import numpy as np
from ultralytics import YOLO

model = YOLO('./data/tire.pt')

def run_tire_test():
    cap = cv2.VideoCapture(0) # add data path here if testing on video
    test_started = False

    print("Test ready.")

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        display_frame = frame.copy()

        # status indicator
        status_text = "STATUS: ACTIVE" if test_started else "STATUS: STATIONARY"
        color = (0, 255, 0) if test_started else (0, 0, 255)
        cv2.putText(display_frame, status_text, (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        if test_started:
            # run inference
            results = model(frame, stream=True)

            for r in results:
                # create a black mask for extracted shape
                extracted_tire_layer = np.zeros_like(frame)

                # check if tire is detected
                if len(r.boxes) > 0:
                    for box in r.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        confidence = box.conf[0]

                        if confidence > 0.5:
                            tire_crop = frame[y1:y2, x1:x2]
                            extracted_tire_layer[y1:y2, x1:x2] = tire_crop

                            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                            cv2.putText(display_frame, f"Tire: {confidence:.2f}", (x1, y1-10), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                            
                cv2.imshow("GUI: Extracted Tire Shape", extracted_tire_layer)

        cv2.imshow("Main Detection", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            test_started = True
            print("Test Started")
        elif key == ord('q'):
            break

    cap.release()

if __name__ == "__main__":
    run_tire_test()

