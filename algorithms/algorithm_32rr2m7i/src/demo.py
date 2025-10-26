import cv2
from hsv import hsv

def main():
    # Initialize the HSV class with a video path
    video_path = "data/comp23_6.MOV"  # Replace with your video file path
    hsv_obj = hsv(video_path)

    hsv_obj.tune("yellow")
    
    # hsv_obj.tune("white")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_path}")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Get the combined mask and individual masks
        combined_mask, masks = hsv_obj.get_mask(frame)

        # Display the original frame, combined mask, and individual masks
        cv2.imshow("Original Frame", frame)
        cv2.imshow("Combined Mask", combined_mask)
        cv2.imshow("Yellow Lane Mask", masks["yellow"])
        cv2.imshow("White Lane Mask", masks["white"])

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()