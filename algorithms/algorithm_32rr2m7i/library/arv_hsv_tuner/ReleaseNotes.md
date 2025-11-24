# Changelog

All notable changes to ARV HSV Tuner will be documented in this file.

## [0.1.2] - 2025-11-23

### Added
- **PySide6 Migration**: 
  - Complete transition from PyQt5 to PySide6 6.10+
  - Better layout handling and improved Apple Silicon support
- **Timeline Scrubber System**:
  - Click-to-seek and drag-to-scrub video navigation
  - Real-time timestamp display (MM:SS or HH:MM:SS format)
  - Automatic position tracking during playback
- **Bookmark/Landmark System**:
  - Press 'T' to bookmark current frame
  - Click bookmarks to jump to specific timestamps
  - Loop mode with adjustable time window (±1 to ±30 seconds)
  - Scrollable bookmark panel with +/- controls
- **YOLO Bounding Box Overlays**:
  - Visual detection boxes on original video frame
  - Lane lines: Green boxes with confidence scores
  - Barrels: Blue boxes with confidence scores
  - Toggle visibility via checkboxes
- **HSV Values Display Panel**: 
  - Shows tuned HSV thresholds for all filters in viewer (to track current calibration)
- **Keyboard Shortcuts**:
  - `T` - Add bookmark at current frame
  - `Space` - Pause/Play toggle
  - `F` or `F11` - Toggle fullscreen
  - `Escape` - Exit fullscreen

### Changed
- **Window Behavior**:
  - You can now either double tap on the window OR:
    - Trigger fullscreen (with F) and then escape.
  - Fixed width panels prevent unwanted resizing when adding bookmarks
- **Playback Speed**: 
  - Now respects native video FPS instead of playing too fast
  - 30 FPS video plays at 30 FPS (previously ~60+ FPS)
  - Timer interval calculated from video metadata
- **FPS Counter**: Now accurately displays actual frame rate (previously showed 2x speed)
- **Tuner Layout**: Landmarks panel moved below Performance widget for better visibility
- **Console Display**: Viewer console auto-expands on startup (60/40 split)

### Fixed
- **YOLO Toggle Spam**: Only logs state changes, not every frame
- **Model Reference**: Auto-aliases `model` to `lane_model` for compatibility
- **Checkbox State Detection**: Fixed PySide6 enum comparison (was always False)
- **Timeline Seek**: Eliminated snapback when clicking timeline slider
- **Window Resize**: Bookmarks no longer cause window to shrink

### Technical
- Replaced `PyQt5` imports with `PySide6` (3 locations per file)
- Changed `app.exec_()` → `app.exec()` (PySide6 syntax)
- Updated enum references: `Qt.Horizontal` → `Qt.Orientation.Horizontal`
- Added `QSizePolicy` and `QScrollArea` for better layout control
- Dependencies: `PySide6>=6.10.0` (replaces `PyQt5>=5.15.0`)


## [0.1.1] - 2025-11-16

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

## [0.1.0] - 2025-11-09

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

### Planned for 1.0.0
- Official release to private artifactory for PIP installs :)
- ~~Landmarks in video to allow user to save problematic spots.~~
- ~~Ability to target landmark with +n, -n time to loop over just the landmark, not the whole video (only for pre-recorded feeds.)~~
- Modularize Qt Viewer, and Tuner.  
  - Ensure decoupling.
  - Comment each block.
  - Provide a design doc on how it interfaces the hsv.py API
  - Run against linter, formatter

### Under Consideration
- [ ] GPU acceleration toggle
  - Maybe it helps to increment FPS if we can extract inverse mask with dedicated GPU, to be investigated.

