# Creating Datasets

[Video tutorial](https://www.youtube.com/watch?v=bLGTZrEHdKs)

1. Take a high quality video going through a course.
2. Convert the video to .mp4 if necessary.
    - [.MOV to .MP4 converter](https://cloudconvert.com/mov-to-mp4)
3. Store the video in DrobBox ".../ML/raw_videos".
4. Place the video in the UMARV-CV repo "/parapeters/input" directory
5. Run the script "/src/scripts/get_frames_from_video.py"
6. Go to [https://app.roboflow.com/umarv-cv](https://app.roboflow.com/umarv-cv)
7. Click "+ New Project" in the top right.
8. Fill out the project details.
    - Project Name = "SP/{dataset_name}"
    - License = "CC BY 4.0"
    - Annotation Group = "Lanes"
    - Project Type = "Show More" -> "Semantic Segmentation"
9. Click "Create Public Project".
    - Don't sync annotations.
10. Drop all the raw images into roboflow by selecting the folder and pointing to "{UMARV-CV repo}/parameters/output/{dataset_name}/data".
11. Once the raw images are exported to roboflow, delete the contents of input and output in the repo parameter folder.
11. Click "Save and Continue".
13. Click "Start Manual Labeling".
14. Add teammates if necessary, max of 3 people including yourself.
15. Assign the images.
16. Add the classes
    1. Click on the "Classes" section on the left.
    2. Type "Lanes, Drivable-Area, Cones" in the text box.
    3. Click "Add Classes".
17. Go to label the images.
    1. Click on the "Annotate" section on the left.
    2. Click "Start Annotating".
18. Label the images.
    1. Use the smart polygon feature.
    2. Click on a lane, cone, or the drivable area.
    3. Every time you click, it auto predicts the label. Add more clicks in other areas of the same class if necessary until the class is fully covered.
    4. Click enter.
    5. Choose which class you are labeling.
    6. Click enter.
    - Tip: You don't need to label every piece of the class at once. You can do them bit by bit. Just hit enter once the focused portion of the class is done then focus to covering another portion of the class. By the end, all portions of the class should be labeled.
    - Tip: Click on "Layers" to see the entire multi-class label.
    - Tip: If after you click on the class, other portions of the image that are not your class become highlighted, click on any part of the covered area and it will remove that portion.
    - Tip: If you made a mistake on a class, you can click on "Layers" and delete the class label that was wrong.
    - Tip: For small labels that are hard to pick up by the smart polygon, you can use the regular polygoon tool.
    - Tip: Pay close attention to detail. Even if they are far or small or in between cracks, any class you recognize, our algorithms and models should too.
19. Once finished annotating all the images. Go through them again for quality assurance to ensure nothing was missed or incorrect.
20. Add the images to a dataset.
    1. Click on the "Annotate" section.
    2. Click on the 100% completed images under the "Annotating" subsection.
    3. Click "Add n images to Dataset"
    4. Make it 100% Train.
    5. Click "Add Images".
21. Generate and download the dataset.
    1. Click on the "Generate" section.
    2. Remove the preprocessing steps in section 3 and click "Continue".
    3. Leave Augmentation empty and click "Continue".
    4. Click "Create".
    5. Click "Download Dataset".
    6. Format = "Semantic Segmentation Masks".
    7. Select "download zip to computer".
    8. Click "Continue".
22. Unpack the dataset.
    1. Unzip the downloaded folder.
    2. Rename the "train" folder to "label".
    3. Move the label folder into the UMARV-CV repo in the "/parapeters/input" directory. Should be "/parameters/input/label/...".
    4. Run the script "/src/scripts/extract_label_from_roboflow.py".
    5. Extract the datset from "/parapeters/output".
    6. Replace "dataset_name" with the dataset name.
23. Export this dataset into the UMARV CV Dropbox in ".../ML/datasets/real_world".
