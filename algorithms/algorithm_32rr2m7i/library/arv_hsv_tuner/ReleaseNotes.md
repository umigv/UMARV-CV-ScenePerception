# Changelog

All notable changes to ARV HSV Tuner will be documented in this file.

## [0.1.1] - 2024-11-16

### Added
- **Image Mode Support** (Thanks https://github.com/adsuri): 
  - `hsv_obj.tune("yellow", image_mode=True)` for single image tuning
  - `hsv_obj.post_processing_all_mask(image_mode=True)` for image review
- **Qt-based Multi-Mask Viewer**: 
  - Grid layout showing original frame, combined mask, and individual color masks (Single image if image mode)
  - Toggle YOLO models (lane lines and barrels) via checkboxes during playback
  - Pause/Play controls for video mode
  - FPS monitoring with color-coded performance indicators
- **YOLO Control System**: 
  - `hsv_obj.set_yolo_usage(lane=True, barrel=False)` to control which models run
  - Improves performance by only running needed models
  - Interactive toggles in viewer UI
- **Detection Tracking**: 
  - Lane line detection counting and display
  - Barrel detection counting and display
  - Real-time detection updates in viewer console
- **Frame Skipping Controls**: Performance optimization slider in Qt tuner
  - Adjustable frame skip (1-5) for faster/slower playback
    - It defaults to 2, but might change to default to native.
  - Helps achieve smoother performance on lower-end hardware

### Changed
- **YOLO Models**: Disabled by default in viewer for better initial performance
  - Users can enable via checkboxes as needed
- **PyQt5 Dependency**: Now automatically installed with the package

### Fixed
- **Qt Tuner Return**: Fixed issue where OpenCV fallback would run after Qt tuner
  - Added explicit `return` statement after successful Qt tuner execution
- **Video/Image Variable Handling**: Corrected variable names in image mode loops
  - `static_image` now used consistently instead of `cap` in image mode

### Documentation
- Added `video_example.py` - Simple video mode example
- Added `image_example.py` - Simple image mode example  


---

## [0.1.0] - 2024-11-09

### Added
- Initial release
- HSV color space tuning for lane detection
- YOLO integration for lane lines and barrel detection
- OpenCV-based GUI tuner
- Multi-filter support (yellow, white, etc.)
- HSV value persistence (JSON storage)
- Morphological operations (erosion, dilution, contour filtering)
- Gamma adjustment support

### Dependencies
- opencv-python >= 4.5.0
- numpy >= 1.19.0
- ultralytics >= 8.0.0
- PyQt5 >= 5.15.0

---

## Future Roadmap

### Planned for 1.1.0
- Official release to private artifactory for PIP installs :)
- Landmarks in video to allow user to save problematic spots.
- Ability to target landmark with +n, -n time to loop over just the landmark, not the whole video (only for pre-recorded feeds.)

### Under Consideration
- [ ] GPU acceleration toggle
  - Maybe it helps to increment FPS if we can extract inverse mask with dedicated GPU, to be investigated.

