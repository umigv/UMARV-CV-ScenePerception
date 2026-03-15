# HSV TUNING
### by Matthew Gawthrop

## Algorithm description
Utilizes hsv filtering and YOLO ML models to gain a consistent occupancy grid of mainly lane line, potholes, and traffic barrels. 

`hsv.py` is designed to be used as a library, meaning the file is meant to be copied into other projects and then imported into a file for use `from hsv import hsv`. Each object of the hsv class is tied to a video path. This video path is not only used for saving of hsv filtering values but also is directly used for streaming video during tuning.

### Tuning
Reference `demo.py` for a look at the functionality with tuning. 

You must first create an object with the video path that you want to tune for. Then by calling `hsv_obj.tune(mask_name)` you can slide the sliders back and forth until you reach the mask you want. Finally, slide the "Done Tuning" slider and the values will be saved in `hsv_values.json`. You can do this over and over with new mask_names to get a whole suite of masks that are connected to one video path. Change the video path on construction of the object and you can create a new set of masks for that video too.

Once you have completed all your tuning, use `get_mask(frame)` to get the individual and combined masks you tuned for.

### Getting Occ Grids
Use `combined, mask_dictionary = get_mask(frame)` which will return a 2D numpy array of all combined masks that have been tuned for this object. It also returns a dictionary which maps from mask_name -> occ grid which can be used to get just one colors occupancy grid at a time. 