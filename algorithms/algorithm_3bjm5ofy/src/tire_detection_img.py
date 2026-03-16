import cv2
import torch
import numpy as np
from ultralytics import YOLO

model = YOLO('./data/tires.pt')

def run_tire_test():
    cap_img = cv2.imread('./data/tireImg1.png') # add data path here if testing on video
    test_started = False

    print("Test ready.")
# Maybe not needed
 #   while cap.isOpened():
 #       ret, frame = cap.read()

  #      if not ret:
   #         break

    display_frame = cap_img.copy()

    # UI Overlay: Status Indicator
    status_text = "STATUS: ACTIVE" if test_started else "STATUS: STATIONARY - WAIT"
    color = (0, 255, 0) if test_started else (0, 0, 255)
    cv2.putText(display_frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    test_started = True 
    
    if test_started:
        # 2. Run Inference
        results = model(cap_img)
        
            # Create a black mask for 'Extracted Shape' requirement
        extracted_tire_layer = np.zeros_like(cap_img)
        
        # get the boxes
        boxes = results[0].boxes
        # Check if any tire is detected
        if len(results[0].boxes) > 0:
            for box in boxes:
                # Get coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0]

                if conf > 0.5: # Confidence threshold
                    # Requirement: "Extracted shape of a tire MUST be present"
                    # We crop the detected tire and place it on the black layer
                    tire_crop = cap_img[y1:y2, x1:x2]
                    extracted_tire_layer[y1:y2, x1:x2] = tire_crop
                    
                    # Draw bounding box on main GUI for "Identification"
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(display_frame, f"Tire: {conf:.2f}", (x1, y1-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Show the Extracted Shape Window (Requirement 3)
            cv2.imshow("GUI: Extracted Tire Shape", extracted_tire_layer)

    # Show Main GUI
    cv2.imshow("Main Detection Feed", display_frame)

    
    key = cv2.waitKey(0) & 0xFF
    if key == ord('s'):
        test_started = True
        print("Test Started")
    elif key == ord('q'):
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run_tire_test()

