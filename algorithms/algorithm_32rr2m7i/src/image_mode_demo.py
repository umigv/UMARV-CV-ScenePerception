import cv2
from hsv import hsv

def main():
  video_path = "data/000001.jpg"
  hsv_obj = hsv(video_path)

  hsv_obj.tune("yellow", True)
  hsv_obj.save_single_range("yellow")

if __name__ == "__main__":
  main()