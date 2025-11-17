"""Example usage of arv_hsv_tuner"""
import cv2
from arv_hsv_tuner import hsv
import time

def main():
    # Initialize with video path
    video_path = "<yourpath>"

    # Pass your models, and the path to save the masks.
    barrel_model_path = "<yourmodel>"
    lane_model_path = "<yourmodel"
    tuned_hsv_values_path = "<your_tuned_hsv_file_name>.json" 
    hsv_obj = hsv(video_path,
                  barrel_model_path=barrel_model_path,lane_model_path=lane_model_path,
                  tuned_hsv_path=tuned_hsv_values_path)

    # Tune filters (Qt GUI will open if PyQt5 is installed)
    hsv_obj.tune("yellow")
    hsv_obj.tune("white")
    
    # To display some values.
    time.sleep(1)
    
    # To review your masks, combined, invidually, and the raw video/image feed.
    # Will also diplay on the bottom a terminal to showcase detection.
    hsv_obj.post_processing_all_mask()


if __name__ == "__main__":
    main()