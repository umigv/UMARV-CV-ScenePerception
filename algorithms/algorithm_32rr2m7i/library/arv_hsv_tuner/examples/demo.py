"""Example usage of arv_hsv_tuner"""
import cv2
from arv_hsv_tuner import hsv
import time

def main():
    # Initialize with video path
    video_path = "data/comp23_6.MOV"
    hsv_obj = hsv(video_path)

    print("=" * 60)
    print("STEP 1: HSV TUNING")
    print("=" * 60)
    print("Adjust the HSV sliders to isolate your target color.")
    print("Click 'Done Tuning' button when satisfied.\n")
    
    # Tune filters (Qt GUI will open if PyQt5 is installed)
    hsv_obj.tune("yellow")
    
    print("\n" + "=" * 60)
    print("STEP 2: VIDEO PROCESSING")
    print("=" * 60)
    print("Now processing video with your tuned HSV values...")
    print("Press 'q' to quit.\n")
    
    # To display some values.
    time.sleep(1)
    
    # Process video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_path}")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Get masks
        combined_mask, masks = hsv_obj.get_mask(frame)

        # Display results
        cv2.imshow("Original Frame", frame)
        cv2.imshow("Combined Mask", combined_mask)
        cv2.imshow("Yellow Lane Mask", masks["yellow"])

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()