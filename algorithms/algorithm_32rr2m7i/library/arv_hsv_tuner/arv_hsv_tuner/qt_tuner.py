"""Qt-based HSV tuning interface - Matches original OpenCV behavior"""
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QSlider, QPushButton, QGroupBox)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
import cv2
import numpy as np
import sys
import platform


class QtTunerWindow(QMainWindow):
    """Qt window for HSV parameter tuning"""
    
    def __init__(self, hsv_instance, filter_name, image_mode=False):
        super().__init__()
        self.hsv = hsv_instance
        self.filter_name = filter_name
        self.image_mode = image_mode  # NEW
        self.cap = None
        self.timer = QTimer()
        
        # State tracking
        self.is_dragging = False
        self.current_frame = None
        self.current_mask = None
        
        # NEW: Frame skipping (only for video mode)
        self.frame_skip = 2
        self.frame_counter = 0
        
        if filter_name not in self.hsv.hsv_filters:
            self.hsv.hsv_filters[filter_name] = {
                'h_upper': 179, 'h_lower': 0,
                's_upper': 255, 's_lower': 0,
                'v_upper': 255, 'v_lower': 0
            }
        
        self.setup_ui()
        self.timer.timeout.connect(self.update_frame)
        
    def setup_ui(self):
        mode_text = "Image" if self.image_mode else "Video"
        self.setWindowTitle(f"ARV HSV Tuner - {self.filter_name} ({mode_text} Mode)")
        self.setGeometry(100, 100, 1400, 800)
        
        # M1 optimization
        if platform.system() == "Darwin":
            self.setAttribute(Qt.WA_NativeWindow, True)
        
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        
        # Left: Controls
        controls = self.create_controls()
        main_layout.addWidget(controls)
        
        # Right: Video displays
        video_layout = QVBoxLayout()
        
        video_layout.addWidget(QLabel("<b>Original Video</b>"))
        
        self.label_video = QLabel("Video")
        self.label_video.setAlignment(Qt.AlignCenter)
        self.label_video.setMinimumSize(640, 360)
        video_layout.addWidget(self.label_video)
        
        video_layout.addWidget(QLabel(f"<b>HSV Mask - {self.filter_name}</b>"))
        
        self.label_mask = QLabel("Mask")
        self.label_mask.setAlignment(Qt.AlignCenter)
        self.label_mask.setMinimumSize(640, 360)
        video_layout.addWidget(self.label_mask)
        
        main_layout.addLayout(video_layout, stretch=1)
        
        # Styling
        self.setStyleSheet("""
            QMainWindow { background-color: #2b2b2b; }
            QLabel { color: white; }
            QLabel[alignment="132"] {
                background-color: #1a1a1a;
                border: 2px solid #555;
                border-radius: 3px;
            }
            QGroupBox { 
                color: white; 
                border: 2px solid #555;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
                font-weight: bold;
            }
            QGroupBox::title { 
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QPushButton {
                background-color: #0d7377;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 4px;
                font-weight: bold;
                min-height: 30px;
            }
            QPushButton:hover { background-color: #14a085; }
            QPushButton:pressed { background-color: #0a5f63; }
            QSlider::groove:horizontal {
                height: 8px;
                background: #555555;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #0d7377;
                width: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
            QSlider::handle:horizontal:hover {
                background: #14a085;
            }
        """)
        
    def create_controls(self):
        widget = QWidget()
        widget.setMaximumWidth(380)
        layout = QVBoxLayout(widget)
        
        # HSV Thresholds Group
        hsv_group = QGroupBox("HSV Thresholds")
        hsv_layout = QVBoxLayout()
        
        self.sliders = {}
        values = self.hsv.hsv_filters[self.filter_name]
        
        for param, max_val in [('h_upper', 179), ('h_lower', 179),
                            ('s_upper', 255), ('s_lower', 255),
                            ('v_upper', 255), ('v_lower', 255)]:
            slider_widget = self.create_slider(
                param.replace('_', ' ').title(),
                0, max_val, values[param], param
            )
            hsv_layout.addWidget(slider_widget)
        
        hsv_group.setLayout(hsv_layout)
        layout.addWidget(hsv_group)
        
        # NEW: Performance Group
        perf_group = QGroupBox("Performance")
        perf_layout = QVBoxLayout()
        
        # Frame skip control
        skip_label = QLabel("Frame Skip (higher = faster, choppier)")
        skip_label.setStyleSheet("color: white; border: none; background: transparent; font-size: 11px;")
        perf_layout.addWidget(skip_label)
        
        skip_widget = QWidget()
        skip_layout = QHBoxLayout(skip_widget)
        skip_layout.setContentsMargins(0, 0, 0, 0)
        
        skip_lbl = QLabel("Skip:")
        skip_lbl.setMinimumWidth(80)
        skip_lbl.setStyleSheet("color: white; border: none; background: transparent;")
        
        self.skip_slider = QSlider(Qt.Horizontal)
        self.skip_slider.setMinimum(1)  # Process every frame
        self.skip_slider.setMaximum(5)  # Skip up to 4 frames
        self.skip_slider.setValue(2)    # Default: skip 1 frame
        
        self.skip_value_label = QLabel("2 (Every 2nd)")
        self.skip_value_label.setMinimumWidth(100)
        self.skip_value_label.setStyleSheet("color: white; border: none; background: transparent;")
        
        self.skip_slider.valueChanged.connect(self.on_skip_changed)
        
        skip_layout.addWidget(skip_lbl)
        skip_layout.addWidget(self.skip_slider, stretch=1)
        skip_layout.addWidget(self.skip_value_label)
        
        perf_layout.addWidget(skip_widget)
        
        # Info text
        info_text = QLabel(
            "• 1 = Every frame (slowest, smoothest)\n"
            "• 2 = Every 2nd frame (recommended)\n"
            "• 3+ = Every 3rd+ frame (faster, choppy)"
        )
        info_text.setStyleSheet("color: #888; border: none; background: transparent; font-size: 10px;")
        info_text.setWordWrap(True)
        perf_layout.addWidget(info_text)
        
        perf_group.setLayout(perf_layout)
        layout.addWidget(perf_group)
        
        layout.addStretch()
        
        # Done button
        btn_done = QPushButton("✓ Done Tuning")
        btn_done.clicked.connect(self.close)
        layout.addWidget(btn_done)
        
        return widget

    def on_skip_changed(self, value):
        """Update frame skip value"""
        self.frame_skip = value
        
        descriptions = {
            1: "1 (Every frame)",
            2: "2 (Every 2nd)",
            3: "3 (Every 3rd)",
            4: "4 (Every 4th)",
            5: "5 (Every 5th)"
        }
        
        self.skip_value_label.setText(descriptions.get(value, str(value)))
        print(f"Frame skip set to {value} - processing every {value}{self.get_ordinal(value)} frame")

    def get_ordinal(self, n):
        """Get ordinal suffix (1st, 2nd, 3rd, etc)"""
        if 10 <= n % 100 <= 20:
            suffix = 'th'
        else:
            suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
        return suffix
    
    def create_slider(self, label, min_val, max_val, initial, param_name):
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 5, 0, 5)
        
        lbl = QLabel(f"{label}:")
        lbl.setMinimumWidth(80)
        lbl.setStyleSheet("color: white; border: none; background: transparent;")
        
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(min_val)
        slider.setMaximum(max_val)
        slider.setValue(initial)
        
        # Connect to different handlers for drag vs release
        slider.sliderPressed.connect(self.on_slider_pressed)
        slider.sliderReleased.connect(self.on_slider_released)
        slider.valueChanged.connect(lambda v: self.on_slider_change(param_name, v))
        
        value_label = QLabel(str(initial))
        value_label.setMinimumWidth(40)
        value_label.setStyleSheet("color: white; border: none; background: transparent;")
        slider.valueChanged.connect(lambda v: value_label.setText(str(v)))
        
        layout.addWidget(lbl)
        layout.addWidget(slider, stretch=1)
        layout.addWidget(value_label)
        
        self.sliders[param_name] = slider
        return widget
    
    def on_slider_pressed(self):
        """Called when user starts dragging a slider"""
        self.is_dragging = True
        # Stop video playback while dragging
        self.timer.stop()
    
    def on_slider_released(self):
        """Called when user releases a slider"""
        self.is_dragging = False
        # Resume video playback
        self.timer.start()
    
    def on_slider_change(self, param, value):
        """Update HSV filter values and redraw"""
        # Update the filter value
        self.hsv.hsv_filters[self.filter_name][param] = value
        
        if self.image_mode:
            # Image mode: Update immediately
            self.display_static_image()
        else:
            # Video mode: Only update if dragging
            if self.is_dragging and self.current_frame is not None:
                self.process_and_display_frozen_frame()
    
    def process_and_display_frozen_frame(self):
        """Process the frozen frame with updated HSV values"""
        # Use the stored frame
        self.hsv.image = self.current_frame.copy()
        
        try:
            self.hsv.adjust_gamma()
        except Exception:
            pass
        
        self.hsv.hsv_image = cv2.cvtColor(self.hsv.image, cv2.COLOR_BGR2HSV)
        
        # Run full pipeline to get updated mask
        try:
            combined_mask, mask_dict = self.hsv.update_mask()
            
            if self.filter_name in mask_dict:
                current_mask = mask_dict[self.filter_name]
            else:
                current_mask = combined_mask if combined_mask is not None else np.zeros_like(self.hsv.hsv_image[:,:,0])
                
        except Exception as e:
            # Fallback
            values = self.hsv.hsv_filters[self.filter_name]
            lower = np.array([values['h_lower'], values['s_lower'], values['v_lower']])
            upper = np.array([values['h_upper'], values['s_upper'], values['v_upper']])
            current_mask = cv2.inRange(self.hsv.hsv_image, lower, upper)
        
        # Update ONLY the mask display (video stays frozen)
        if len(current_mask.shape) == 2:
            mask_bgr = cv2.cvtColor(current_mask, cv2.COLOR_GRAY2BGR)
        else:
            mask_bgr = current_mask
        
        self.display_image(mask_bgr, self.label_mask)
        self.current_mask = current_mask
    
    def start(self):
        """Initialize video/image capture"""
        if self.image_mode:
            # Load single image
            frame = cv2.imread(self.hsv.video_path)
            if frame is None:
                print(f"Error: Unable to open image file {self.hsv.video_path}")
                return False
            
            self.current_frame = frame
            print(f"Image loaded: {self.hsv.video_path}")
            print(f"Image size: {frame.shape[1]}x{frame.shape[0]}")
            print(f"Image mode: Static image tuning")
            
            self.hsv.setup = True
            
            # Display the image once
            self.display_static_image()
            
            # No timer needed for static image - only update on slider change
            return True
        else:
            # Video mode (existing code)
            self.cap = cv2.VideoCapture(self.hsv.video_path)
            if not self.cap.isOpened():
                print(f"Error: Unable to open video file {self.hsv.video_path}")
                return False
            
            native_fps = self.cap.get(cv2.CAP_PROP_FPS)
            if native_fps <= 0 or native_fps > 240:
                native_fps = 30
            
            print(f"Video loaded: {self.hsv.video_path}")
            print(f"Native FPS: {native_fps:.1f}")
            print(f"Running full pipeline (HSV + morphology + YOLO)")
            
            self.hsv.setup = True
            
            # Start timer at 30 FPS
            self.timer.start(33)
            
            return True
        
    def display_static_image(self):
        """Display static image with current HSV settings"""
        frame = self.current_frame.copy()
        
        # Process image
        self.hsv.image = frame
        
        try:
            self.hsv.adjust_gamma()
        except Exception as e:
            print(f"Warning: adjust_gamma failed: {e}")
        
        self.hsv.hsv_image = cv2.cvtColor(self.hsv.image, cv2.COLOR_BGR2HSV)
        
        # Get mask
        try:
            combined_mask, mask_dict = self.hsv.update_mask()
            
            if self.filter_name in mask_dict:
                current_mask = mask_dict[self.filter_name]
            else:
                current_mask = combined_mask if combined_mask is not None else np.zeros_like(self.hsv.hsv_image[:,:,0])
                
        except Exception as e:
            print(f"Error in update_mask: {e}")
            values = self.hsv.hsv_filters[self.filter_name]
            lower = np.array([values['h_lower'], values['s_lower'], values['v_lower']])
            upper = np.array([values['h_upper'], values['s_upper'], values['v_upper']])
            current_mask = cv2.inRange(self.hsv.hsv_image, lower, upper)
        
        # Display
        self.display_image(frame, self.label_video)
        
        if len(current_mask.shape) == 2:
            mask_bgr = cv2.cvtColor(current_mask, cv2.COLOR_GRAY2BGR)
        else:
            mask_bgr = current_mask
        self.display_image(mask_bgr, self.label_mask)
    
    def update_frame(self):
        """Process and display video frame with frame skipping"""
        if self.image_mode:
            return  # No-op for image mode
        
        # Increment frame counter
        self.frame_counter += 1
        
        # Skip frames based on frame_skip setting
        if self.frame_counter % self.frame_skip != 0:
            # Still need to read the frame to advance video
            if not self.is_dragging:
                ret, _ = self.cap.read()
                if not ret:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.frame_counter = 0
            return  # Skip processing/display
        
        # Only update video if not dragging
        if self.is_dragging:
            return
        
        ret, frame = self.cap.read()
        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.frame_counter = 0
            ret, frame = self.cap.read()
            if not ret:
                return
        
        # Store the current frame for use during slider dragging
        self.current_frame = frame.copy()
        
        # Process frame
        self.hsv.image = frame
        
        try:
            self.hsv.adjust_gamma()
        except Exception as e:
            print(f"Warning: adjust_gamma failed: {e}")
        
        self.hsv.hsv_image = cv2.cvtColor(self.hsv.image, cv2.COLOR_BGR2HSV)
        
        # Call YOUR update_mask (full pipeline)
        try:
            combined_mask, mask_dict = self.hsv.update_mask()
            
            if self.filter_name in mask_dict:
                current_mask = mask_dict[self.filter_name]
            else:
                current_mask = combined_mask if combined_mask is not None else np.zeros_like(self.hsv.hsv_image[:,:,0])
                    
        except Exception as e:
            print(f"Error in update_mask: {e}")
            values = self.hsv.hsv_filters[self.filter_name]
            lower = np.array([values['h_lower'], values['s_lower'], values['v_lower']])
            upper = np.array([values['h_upper'], values['s_upper'], values['v_upper']])
            current_mask = cv2.inRange(self.hsv.hsv_image, lower, upper)
        
        # Store current mask
        self.current_mask = current_mask
        
        # Display both video and mask
        self.display_image(frame, self.label_video)
        
        if len(current_mask.shape) == 2:
            mask_bgr = cv2.cvtColor(current_mask, cv2.COLOR_GRAY2BGR)
        else:
            mask_bgr = current_mask
        self.display_image(mask_bgr, self.label_mask)
    
    def display_image(self, img, label):
        """Convert OpenCV image to Qt and display - Optimized"""
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        # Resize with OpenCV before Qt conversion (faster)
        target_size = label.size()
        target_width = target_size.width()
        target_height = target_size.height()
        
        if target_width > 0 and target_height > 0:
            h, w = img.shape[:2]
            aspect = w / h
            target_aspect = target_width / target_height
            
            if aspect > target_aspect:
                new_w = target_width
                new_h = int(new_w / aspect)
            else:
                new_h = target_height
                new_w = int(new_h * aspect)
            
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Convert to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = img_rgb.shape
        
        # Ensure contiguous
        img_rgb = np.ascontiguousarray(img_rgb)
        
        # Create QImage and display
        qt_image = QImage(img_rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        label.setPixmap(pixmap)
    
    def closeEvent(self, event):
        """Cleanup when window closes"""
        self.timer.stop()
        if self.cap and not self.image_mode:
            self.cap.release()
        self.hsv.setup = False
        
        try:
            self.hsv.save_hsv_values()
            print(f"\n✓ HSV values saved for filter '{self.filter_name}'")
            print(f"  Values: {self.hsv.hsv_filters[self.filter_name]}")
        except Exception as e:
            print(f"Warning: Could not save HSV values: {e}")
        
        event.accept()


def tune_with_qt(hsv_instance, filter_name, image_mode=False):
    """
    Launch Qt-based tuning interface
    
    Args:
        hsv_instance: Instance of hsv class
        filter_name: Name of filter to tune
        image_mode: If True, tune on static image instead of video
    
    Behavior:
    - Video mode: Video plays and loops, freezes while dragging sliders
    - Image mode: Static image, updates immediately on slider change
    """
    import os
    
    # M1 Mac optimizations
    if platform.system() == "Darwin":
        os.environ["QT_MAC_WANTS_LAYER"] = "1"
        os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "0"
    
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
        
        if platform.system() == "Darwin":
            app.setAttribute(Qt.AA_UseHighDpiPixmaps, False)
    
    window = QtTunerWindow(hsv_instance, filter_name, image_mode=image_mode)
    if window.start():
        window.show()
        app.exec_()