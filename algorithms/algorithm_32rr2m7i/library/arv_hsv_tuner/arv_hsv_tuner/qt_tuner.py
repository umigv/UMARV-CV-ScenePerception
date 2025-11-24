"""Qt-based HSV tuning interface - Matches original OpenCV behavior"""
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QSlider, QPushButton, QGroupBox,
                             QSizePolicy)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
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
        self.image_mode = image_mode
        self.cap = None
        self.timer = QTimer()
        
        # State tracking
        self.is_dragging = False
        self.is_paused = False  # For pause/play
        self.current_frame = None
        self.current_mask = None
        
        # Video timeline tracking
        self.total_frames = 0
        self.current_frame_pos = 0
        self.video_fps = 30
        self.is_seeking = False
        
        # Landmark/bookmark system
        self.landmarks = []  # List of {frame: int, time: float, window: int}
        self.active_landmark = None  # Currently selected landmark for looping
        self.loop_start_frame = None
        self.loop_end_frame = None
        
        # YOLO detection toggles
        self.show_lane_detections = False
        self.show_barrel_detections = False
        self.detection_log = []  # Store recent detections
        self.max_log_entries = 100
        
        # Frame skipping (only for video mode)
        self.frame_skip = 1
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
        
        # Get available screen geometry (excludes dock/menubar)
        screen = QApplication.primaryScreen()
        screen_geometry = screen.availableGeometry()
        
        # Set window to fill available space
        self.setGeometry(screen_geometry)
        
        # Prevent automatic resizing
        self.setSizePolicy(
            QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        )
        
        # M1 optimization
        if platform.system() == "Darwin":
            self.setAttribute(Qt.WA_NativeWindow, True)
        
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        
        # Left: Controls (includes landmarks now)
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
        
        # Timeline controls (only for video mode)
        if not self.image_mode:
            timeline_group = self.create_timeline_controls()
            video_layout.addWidget(timeline_group)
            
            # Detection console and YOLO controls
            detection_group = self.create_detection_controls()
            video_layout.addWidget(detection_group)
        
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
    
    def keyPressEvent(self, event):
        """Handle keyboard shortcuts"""
        if event.key() == Qt.Key.Key_T:
            if not self.image_mode:
                self.add_landmark()
        elif event.key() == Qt.Key.Key_Space:
            if not self.image_mode:
                self.toggle_pause()
        elif event.key() == Qt.Key.Key_F or (event.key() == Qt.Key.Key_F11):
            # Toggle fullscreen with F or F11
            self.toggle_fullscreen()
        elif event.key() == Qt.Key.Key_Escape:
            # Exit fullscreen with Escape
            if self.isFullScreen():
                self.showMaximized()
        else:
            super().keyPressEvent(event)
    
    def toggle_fullscreen(self):
        """Toggle between fullscreen and maximized"""
        if self.isFullScreen():
            self.showMaximized()
        else:
            self.showFullScreen()
    
    def create_landmarks_panel(self):
        """Create landmarks/bookmarks panel for quick navigation"""
        panel = QGroupBox("Landmarks (Press 'T')")
        panel.setMaximumWidth(250)
        panel.setMinimumWidth(250)  # Fixed width
        
        # Prevent panel from resizing parent
        panel.setSizePolicy(
            QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        )
        
        layout = QVBoxLayout()
        
        # Instructions
        info = QLabel("Press 'T' to bookmark\ncurrent frame")
        info.setStyleSheet("color: #aaa; font-size: 11px; font-weight: normal;")
        layout.addWidget(info)
        
        # Scrollable area for landmarks
        from PySide6.QtWidgets import QScrollArea
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: transparent;
            }
        """)
        
        # Landmarks list container
        self.landmarks_widget = QWidget()
        self.landmarks_layout = QVBoxLayout(self.landmarks_widget)
        self.landmarks_layout.setSpacing(5)
        self.landmarks_layout.addStretch()
        
        scroll.setWidget(self.landmarks_widget)
        layout.addWidget(scroll)
        
        panel.setLayout(layout)
        return panel
    
    def add_landmark(self):
        """Add a landmark at current position"""
        if self.cap is None or self.total_frames == 0:
            return
        
        frame_pos = self.current_frame_pos
        time_sec = frame_pos / self.video_fps
        
        # Check if landmark already exists at this position (within 1 second)
        for lm in self.landmarks:
            if abs(lm['frame'] - frame_pos) < self.video_fps:
                print(f"Landmark already exists near {self.format_time(time_sec)}")
                return
        
        landmark = {
            'frame': frame_pos,
            'time': time_sec,
            'window': 3  # Default ±3 seconds
        }
        self.landmarks.append(landmark)
        
        # Create widget for this landmark
        self.create_landmark_widget(len(self.landmarks) - 1, landmark)
        
        print(f"✓ Landmark added at {self.format_time(time_sec)}")
    
    def create_landmark_widget(self, index, landmark):
        """Create UI widget for a landmark"""
        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(5, 5, 5, 5)
        container_layout.setSpacing(3)
        
        # Top row: Time + Delete button
        top_row = QHBoxLayout()
        
        time_btn = QPushButton(f"⏱ {self.format_time(landmark['time'])}")
        time_btn.setStyleSheet("""
            QPushButton {
                background-color: #1a4d2e;
                color: white;
                border: none;
                padding: 5px;
                text-align: left;
                font-size: 12px;
            }
            QPushButton:hover { background-color: #2d6a4f; }
            QPushButton:pressed { background-color: #0d3320; }
        """)
        time_btn.clicked.connect(lambda: self.jump_to_landmark(index))
        top_row.addWidget(time_btn)
        
        del_btn = QPushButton("✕")
        del_btn.setMaximumWidth(30)
        del_btn.setStyleSheet("""
            QPushButton {
                background-color: #8b0000;
                color: white;
                border: none;
                padding: 3px;
            }
            QPushButton:hover { background-color: #a00000; }
        """)
        del_btn.clicked.connect(lambda: self.delete_landmark(index))
        top_row.addWidget(del_btn)
        
        container_layout.addLayout(top_row)
        
        # Loop controls row
        loop_row = QHBoxLayout()
        
        loop_btn = QPushButton("⟲ Loop")
        loop_btn.setCheckable(True)
        loop_btn.setMaximumWidth(70)
        loop_btn.setStyleSheet("""
            QPushButton {
                background-color: #4a4a4a;
                color: white;
                border: none;
                padding: 3px;
                font-size: 10px;
            }
            QPushButton:checked {
                background-color: #0d7377;
            }
            QPushButton:hover { background-color: #5a5a5a; }
        """)
        loop_btn.clicked.connect(lambda checked: self.toggle_landmark_loop(index, checked))
        loop_row.addWidget(loop_btn)
        
        # Window size controls
        window_label = QLabel(f"±{landmark['window']}s")
        window_label.setStyleSheet("color: white; font-size: 10px;")
        window_label.setMinimumWidth(40)
        loop_row.addWidget(window_label)
        
        minus_btn = QPushButton("-")
        minus_btn.setMaximumWidth(25)
        minus_btn.clicked.connect(lambda: self.adjust_landmark_window(index, -1, window_label))
        loop_row.addWidget(minus_btn)
        
        plus_btn = QPushButton("+")
        plus_btn.setMaximumWidth(25)
        plus_btn.clicked.connect(lambda: self.adjust_landmark_window(index, 1, window_label))
        loop_row.addWidget(plus_btn)
        
        container_layout.addLayout(loop_row)
        
        # Store references
        landmark['widget'] = container
        landmark['loop_btn'] = loop_btn
        landmark['window_label'] = window_label
        
        # Add to layout (before the stretch)
        self.landmarks_layout.insertWidget(len(self.landmarks) - 1, container)
    
    def jump_to_landmark(self, index):
        """Jump to a specific landmark"""
        if index >= len(self.landmarks):
            return
        
        landmark = self.landmarks[index]
        frame_pos = landmark['frame']
        
        # Pause and seek
        was_playing = not self.is_paused
        if was_playing:
            self.toggle_pause()
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
        self.current_frame_pos = frame_pos
        
        # Update slider
        if self.total_frames > 0:
            timeline_value = int((frame_pos / self.total_frames) * 1000)
            self.timeline_slider.blockSignals(True)
            self.timeline_slider.setValue(timeline_value)
            self.timeline_slider.blockSignals(False)
        
        # Display frame
        ret, frame = self.cap.read()
        if ret:
            self.current_frame = frame.copy()
            self.seek_and_display_frame()
        
        self.update_time_display()
        print(f"→ Jumped to landmark at {self.format_time(landmark['time'])}")
    
    def toggle_landmark_loop(self, index, checked):
        """Toggle loop mode for a landmark"""
        if index >= len(self.landmarks):
            return
        
        landmark = self.landmarks[index]
        
        if checked:
            # Deactivate other landmarks
            for i, lm in enumerate(self.landmarks):
                if i != index and 'loop_btn' in lm:
                    lm['loop_btn'].setChecked(False)
            
            # Activate this landmark
            self.active_landmark = index
            window_frames = int(landmark['window'] * self.video_fps)
            self.loop_start_frame = max(0, landmark['frame'] - window_frames)
            self.loop_end_frame = min(self.total_frames - 1, landmark['frame'] + window_frames)
            
            print(f"⟲ Loop mode: {self.format_time(self.loop_start_frame / self.video_fps)} - {self.format_time(self.loop_end_frame / self.video_fps)}")
            
            # Jump to start of loop
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.loop_start_frame)
            self.current_frame_pos = self.loop_start_frame
            
            # Start playing if paused
            if self.is_paused:
                self.toggle_pause()
        else:
            # Deactivate loop
            self.active_landmark = None
            self.loop_start_frame = None
            self.loop_end_frame = None
            print("⟲ Loop mode disabled")
    
    def adjust_landmark_window(self, index, delta, label):
        """Adjust the time window for a landmark"""
        if index >= len(self.landmarks):
            return
        
        landmark = self.landmarks[index]
        landmark['window'] = max(1, min(30, landmark['window'] + delta))
        label.setText(f"±{landmark['window']}s")
        
        # Update loop if this landmark is active
        if self.active_landmark == index:
            window_frames = int(landmark['window'] * self.video_fps)
            self.loop_start_frame = max(0, landmark['frame'] - window_frames)
            self.loop_end_frame = min(self.total_frames - 1, landmark['frame'] + window_frames)
    
    def delete_landmark(self, index):
        """Delete a landmark"""
        if index >= len(self.landmarks):
            return
        
        landmark = self.landmarks[index]
        
        # Deactivate if active
        if self.active_landmark == index:
            self.active_landmark = None
            self.loop_start_frame = None
            self.loop_end_frame = None
        
        # Remove widget
        if 'widget' in landmark:
            self.landmarks_layout.removeWidget(landmark['widget'])
            landmark['widget'].deleteLater()
        
        # Remove from list
        self.landmarks.pop(index)
        
        # Update indices for remaining landmarks
        if self.active_landmark is not None and self.active_landmark > index:
            self.active_landmark -= 1
        
        print(f"✕ Landmark deleted")
    
    def create_timeline_controls(self):
        """Create video timeline scrubber and playback controls"""
        timeline_group = QGroupBox("Video Timeline")
        timeline_layout = QVBoxLayout()
        
        # Playback controls row
        controls_layout = QHBoxLayout()
        
        # Pause/Play button
        self.btn_pause = QPushButton("⏸ Pause")
        self.btn_pause.setMaximumWidth(100)
        self.btn_pause.clicked.connect(self.toggle_pause)
        controls_layout.addWidget(self.btn_pause)
        
        # Current time label
        self.time_label = QLabel("00:00 / 00:00")
        self.time_label.setStyleSheet("color: white; font-size: 12px;")
        controls_layout.addWidget(self.time_label)
        
        controls_layout.addStretch()
        timeline_layout.addLayout(controls_layout)
        
        # Timeline slider
        self.timeline_slider = QSlider(Qt.Orientation.Horizontal)
        self.timeline_slider.setMinimum(0)
        self.timeline_slider.setMaximum(1000)
        self.timeline_slider.setValue(0)
        
        # Connect timeline events
        self.timeline_slider.sliderPressed.connect(self.on_timeline_pressed)
        self.timeline_slider.sliderReleased.connect(self.on_timeline_released)
        self.timeline_slider.sliderMoved.connect(self.on_timeline_moved)  # For dragging
        self.timeline_slider.valueChanged.connect(self.on_timeline_clicked)  # For clicking
        
        timeline_layout.addWidget(self.timeline_slider)
        
        timeline_group.setLayout(timeline_layout)
        return timeline_group
    
    def toggle_pause(self):
        """Toggle pause/play for video"""
        if self.is_paused:
            self.timer.start()
            self.btn_pause.setText("⏸ Pause")
            self.is_paused = False
        else:
            self.timer.stop()
            self.btn_pause.setText("▶ Play")
            self.is_paused = True
    
    def on_timeline_pressed(self):
        """User started dragging timeline"""
        self.is_seeking = True
        self.timer.stop()
    
    def on_timeline_released(self):
        """User released timeline"""
        self.is_seeking = False
        
        # Important: Update current_frame_pos to match where we seeked to
        if self.cap is not None:
            self.current_frame_pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        
        # Only resume timer if not paused
        if not self.is_paused:
            self.timer.start()
    
    def on_timeline_moved(self, value):
        """Timeline slider dragged - seek to position and show both views"""
        if self.cap is not None and self.total_frames > 0:
            # Calculate frame position
            frame_pos = int((value / 1000.0) * self.total_frames)
            frame_pos = max(0, min(frame_pos, self.total_frames - 1))  # Clamp to valid range
            
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
            self.current_frame_pos = frame_pos
            
            # Read and display the frame
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame.copy()
                self.seek_and_display_frame()  # Show BOTH views
            
            # Update time display
            self.update_time_display()
    
    def on_timeline_clicked(self, value):
        """Timeline slider clicked (not dragged) - seek to position"""
        # Only handle clicks when not actively seeking (prevents interference with drag)
        if not self.is_seeking and self.cap is not None and self.total_frames > 0:
            # Pause the timer momentarily
            was_paused = self.is_paused
            self.timer.stop()
            
            # Calculate frame position
            frame_pos = int((value / 1000.0) * self.total_frames)
            frame_pos = max(0, min(frame_pos, self.total_frames - 1))
            
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
            self.current_frame_pos = frame_pos
            
            # Read and display the frame
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame.copy()
                self.seek_and_display_frame()
            
            # Update time display
            self.update_time_display()
            
            # Resume timer if wasn't paused
            if not was_paused:
                self.timer.start()
    
    def update_time_display(self):
        """Update the time label with current position"""
        if self.cap is None:
            return
        
        current_sec = self.current_frame_pos / self.video_fps
        total_sec = self.total_frames / self.video_fps
        
        current_time = self.format_time(current_sec)
        total_time = self.format_time(total_sec)
        
        self.time_label.setText(f"{current_time} / {total_time}")
    
    def format_time(self, seconds):
        """Format seconds as HH:MM:SS or MM:SS"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes:02d}:{secs:02d}"
        
    def create_detection_controls(self):
        """Create YOLO detection toggles and console"""
        group = QGroupBox("Detection Console")
        layout = QVBoxLayout()
        
        # YOLO model toggles
        toggles_layout = QHBoxLayout()
        
        self.lane_toggle = QPushButton("🚗 Lane Lines")
        self.lane_toggle.setCheckable(True)
        self.lane_toggle.setChecked(False)
        self.lane_toggle.clicked.connect(self.toggle_lane_detection)
        self.lane_toggle.setStyleSheet("""
            QPushButton {
                background-color: #4a4a4a;
                color: white;
                border: none;
                padding: 8px;
                font-size: 12px;
            }
            QPushButton:checked {
                background-color: #0d7377;
            }
            QPushButton:hover { background-color: #5a5a5a; }
        """)
        toggles_layout.addWidget(self.lane_toggle)
        
        self.barrel_toggle = QPushButton("🛢️ Barrels")
        self.barrel_toggle.setCheckable(True)
        self.barrel_toggle.setChecked(False)
        self.barrel_toggle.clicked.connect(self.toggle_barrel_detection)
        self.barrel_toggle.setStyleSheet("""
            QPushButton {
                background-color: #4a4a4a;
                color: white;
                border: none;
                padding: 8px;
                font-size: 12px;
            }
            QPushButton:checked {
                background-color: #0d7377;
            }
            QPushButton:hover { background-color: #5a5a5a; }
        """)
        toggles_layout.addWidget(self.barrel_toggle)
        
        clear_btn = QPushButton("🗑️ Clear")
        clear_btn.clicked.connect(self.clear_detection_log)
        clear_btn.setMaximumWidth(80)
        clear_btn.setStyleSheet("""
            QPushButton {
                background-color: #8b0000;
                color: white;
                border: none;
                padding: 8px;
                font-size: 11px;
            }
            QPushButton:hover { background-color: #a00000; }
        """)
        toggles_layout.addWidget(clear_btn)
        
        layout.addLayout(toggles_layout)
        
        # Detection console (text area)
        from PySide6.QtWidgets import QTextEdit
        self.detection_console = QTextEdit()
        self.detection_console.setReadOnly(True)
        self.detection_console.setMaximumHeight(150)
        self.detection_console.setStyleSheet("""
            QTextEdit {
                background-color: #1a1a1a;
                color: #00ff00;
                border: 2px solid #555;
                border-radius: 3px;
                font-family: 'Courier New', monospace;
                font-size: 11px;
                padding: 5px;
            }
        """)
        self.detection_console.setPlaceholderText("Detection log will appear here...")
        layout.addWidget(self.detection_console)
        
        group.setLayout(layout)
        return group
    
    def toggle_lane_detection(self, checked):
        """Toggle lane line detection"""
        self.show_lane_detections = checked
        if checked:
            self.log_detection("🚗 Lane detection ENABLED")
        else:
            self.log_detection("🚗 Lane detection DISABLED")
    
    def toggle_barrel_detection(self, checked):
        """Toggle barrel detection"""
        self.show_barrel_detections = checked
        if checked:
            self.log_detection("🛢️ Barrel detection ENABLED")
        else:
            self.log_detection("🛢️ Barrel detection DISABLED")
    
    def clear_detection_log(self):
        """Clear the detection console"""
        self.detection_log.clear()
        self.detection_console.clear()
    
    def log_detection(self, message):
        """Add a detection message to the console"""
        import time
        timestamp = time.strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        
        # Add to log
        self.detection_log.append(log_entry)
        if len(self.detection_log) > self.max_log_entries:
            self.detection_log.pop(0)
        
        # Update console
        self.detection_console.append(log_entry)
        
        # Auto-scroll to bottom
        scrollbar = self.detection_console.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        
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
        
        # Performance Group (only for video mode)
        if not self.image_mode:
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
            self.skip_slider.setMinimum(1)
            self.skip_slider.setMaximum(5)
            self.skip_slider.setValue(1)
            
            self.skip_value_label = QLabel("1 (Every frame)")
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
            
            # Landmarks panel below Performance (only for video mode)
            landmarks_panel = self.create_landmarks_panel()
            layout.addWidget(landmarks_panel)
        
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
        # Resume video playback only if not paused
        if not self.is_paused:
            self.timer.start()
    
    def on_slider_change(self, param, value):
        """Update HSV filter values and redraw"""
        # Update the filter value
        self.hsv.hsv_filters[self.filter_name][param] = value
        
        # Only process if we have a frame loaded
        if self.current_frame is None:
            return
        
        if self.image_mode:
            # Image mode: Update immediately
            self.display_static_image()
        else:
            # Video mode: Only update if dragging
            if self.is_dragging:
                self.process_and_display_frozen_frame()
    
    def seek_and_display_frame(self):
        """Display both original frame and mask during seeking with YOLO overlays"""
        if self.current_frame is None:
            return
        
        frame = self.current_frame.copy()
        
        # Process frame with current HSV settings
        self.hsv.image = frame
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Get current slider values from hsv_filters
        values = self.hsv.hsv_filters[self.filter_name]
        lower = np.array([values['h_lower'], values['s_lower'], values['v_lower']])
        upper = np.array([values['h_upper'], values['s_upper'], values['v_upper']])
        mask = cv2.inRange(hsv_frame, lower, upper)
        
        # Store for potential use
        self.current_mask = mask
        
        # Create display frame for YOLO overlays
        display_frame = frame.copy()
        
        # Run YOLO detections if enabled (but don't log during seeking - too spammy)
        if self.show_lane_detections or self.show_barrel_detections:
            # Run lane detection
            if self.show_lane_detections and hasattr(self.hsv, 'lane_model') and self.hsv.lane_model is not None:
                try:
                    lane_results = self.hsv.lane_model(frame, conf=0.5, verbose=False)
                    if len(lane_results) > 0 and len(lane_results[0].boxes) > 0:
                        # Draw lane detections on display frame
                        for box in lane_results[0].boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            conf = float(box.conf[0])
                            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(display_frame, f"Lane {conf:.2f}", (x1, y1-10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                except Exception:
                    pass
            
            # Run barrel detection
            if self.show_barrel_detections and hasattr(self.hsv, 'barrel_model') and self.hsv.barrel_model is not None:
                try:
                    barrel_results = self.hsv.barrel_model(frame, conf=0.5, verbose=False)
                    if len(barrel_results) > 0 and len(barrel_results[0].boxes) > 0:
                        # Draw barrel detections on display frame
                        for box in barrel_results[0].boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            conf = float(box.conf[0])
                            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                            cv2.putText(display_frame, f"Barrel {conf:.2f}", (x1, y1-10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                except Exception:
                    pass
        
        # Display original frame (with YOLO overlays)
        display_resized = cv2.resize(display_frame, (640, 360))
        rgb_frame = cv2.cvtColor(display_resized, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        self.label_video.setPixmap(QPixmap.fromImage(qt_image))
        
        # Display mask
        display_mask = cv2.resize(mask, (640, 360))
        rgb_mask = cv2.cvtColor(display_mask, cv2.COLOR_GRAY2RGB)
        h, w, ch = rgb_mask.shape
        bytes_per_line = ch * w
        qt_image_mask = QImage(rgb_mask.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        self.label_mask.setPixmap(QPixmap.fromImage(qt_image_mask))
    
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
            # Video mode
            self.cap = cv2.VideoCapture(self.hsv.video_path)
            if not self.cap.isOpened():
                print(f"Error: Unable to open video file {self.hsv.video_path}")
                return False
            
            self.video_fps = self.cap.get(cv2.CAP_PROP_FPS)
            if self.video_fps <= 0 or self.video_fps > 240:
                self.video_fps = 30
            
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Initialize timeline slider
            if hasattr(self, 'timeline_slider'):
                self.timeline_slider.setMaximum(1000)
            
            print(f"Video loaded: {self.hsv.video_path}")
            print(f"Native FPS: {self.video_fps:.1f}")
            print(f"Total frames: {self.total_frames}")
            if self.total_frames > 0:
                duration = self.total_frames / self.video_fps
                print(f"Duration: {self.format_time(duration)}")
            print(f"Running full pipeline (HSV + morphology + YOLO)")
            
            # Check YOLO model availability
            print(f"\n--- YOLO Model Debug ---")
            print(f"HSV class attributes: {[attr for attr in dir(self.hsv) if 'model' in attr.lower()]}")
            
            if hasattr(self.hsv, 'lane_model_path'):
                print(f"Lane model PATH exists: {self.hsv.lane_model_path}")
            if hasattr(self.hsv, 'lane_model') and self.hsv.lane_model is not None:
                print(f"✓ Lane detection model LOADED")
            else:
                print(f"✗ Lane detection model NOT loaded")
                if hasattr(self.hsv, 'lane_model'):
                    print(f"  - lane_model attribute exists but is: {self.hsv.lane_model}")
                else:
                    print(f"  - lane_model attribute doesn't exist")
            
            if hasattr(self.hsv, 'barrel_model_path'):
                print(f"Barrel model PATH exists: {self.hsv.barrel_model_path}")
            if hasattr(self.hsv, 'barrel_model') and self.hsv.barrel_model is not None:
                print(f"✓ Barrel detection model LOADED")
            else:
                print(f"✗ Barrel detection model NOT loaded")
            print(f"--- End Debug ---\n")
            
            # Auto-load models if paths exist but models aren't loaded
            if hasattr(self.hsv, 'lane_model_path') and self.hsv.lane_model_path:
                if not hasattr(self.hsv, 'lane_model') or self.hsv.lane_model is None:
                    try:
                        from ultralytics import YOLO
                        print(f"🔄 Auto-loading lane model from: {self.hsv.lane_model_path}")
                        self.hsv.lane_model = YOLO(self.hsv.lane_model_path)
                        print(f"✓ Lane model loaded successfully!")
                    except Exception as e:
                        print(f"✗ Failed to load lane model: {e}")
                        self.hsv.lane_model = None
            
            # Fallback: Check if lane model is stored as 'model' instead of 'lane_model'
            if hasattr(self.hsv, 'model') and not hasattr(self.hsv, 'lane_model'):
                print(f"⚠️  Found 'model' attribute, aliasing to 'lane_model'")
                self.hsv.lane_model = self.hsv.model
            
            if hasattr(self.hsv, 'barrel_model_path') and self.hsv.barrel_model_path:
                if not hasattr(self.hsv, 'barrel_model') or self.hsv.barrel_model is None:
                    try:
                        from ultralytics import YOLO
                        print(f"🔄 Auto-loading barrel model from: {self.hsv.barrel_model_path}")
                        self.hsv.barrel_model = YOLO(self.hsv.barrel_model_path)
                        print(f"✓ Barrel model loaded successfully!")
                    except Exception as e:
                        print(f"✗ Failed to load barrel model: {e}")
                        self.hsv.barrel_model = None
            
            self.hsv.setup = True
            
            # Calculate timer interval based on native FPS
            timer_interval = int(1000 / self.video_fps)  # milliseconds per frame
            print(f"Timer interval: {timer_interval}ms ({self.video_fps:.1f} FPS)")
            
            self.timer.start(timer_interval)
            
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
        
        # Don't update if seeking or paused
        if self.is_seeking or self.is_paused:
            return
        
        # Increment frame counter
        self.frame_counter += 1
        
        # Skip frames based on frame_skip setting
        if self.frame_skip > 1 and self.frame_counter % self.frame_skip != 0:
            # Still need to read the frame to advance video
            if not self.is_dragging:
                ret, _ = self.cap.read()
                if not ret:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.frame_counter = 0
                    self.current_frame_pos = 0
            return  # Skip processing/display
        
        # Only update video if not dragging
        if self.is_dragging:
            return
        
        ret, frame = self.cap.read()
        if not ret:
            # Check if we're in loop mode
            if self.loop_start_frame is not None:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.loop_start_frame)
                self.current_frame_pos = self.loop_start_frame
            else:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.frame_counter = 0
                self.current_frame_pos = 0
            ret, frame = self.cap.read()
            if not ret:
                return
        
        # Update current position
        self.current_frame_pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        
        # Check if we've exceeded loop bounds
        if self.loop_end_frame is not None and self.current_frame_pos >= self.loop_end_frame:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.loop_start_frame)
            self.current_frame_pos = self.loop_start_frame
            ret, frame = self.cap.read()
            if not ret:
                return
        
        # Update timeline slider (without triggering valueChanged)
        if hasattr(self, 'timeline_slider') and self.total_frames > 0:
            timeline_value = int((self.current_frame_pos / self.total_frames) * 1000)
            self.timeline_slider.blockSignals(True)
            self.timeline_slider.setValue(timeline_value)
            self.timeline_slider.blockSignals(False)
            self.update_time_display()
        
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
        
        # Create display frame for YOLO overlays
        display_frame = frame.copy()
        
        # Run YOLO detections if enabled
        if self.show_lane_detections or self.show_barrel_detections:
            # Run lane detection
            if self.show_lane_detections and hasattr(self.hsv, 'lane_model') and self.hsv.lane_model is not None:
                try:
                    lane_results = self.hsv.lane_model(frame, conf=0.5, verbose=False)
                    if len(lane_results) > 0 and len(lane_results[0].boxes) > 0:
                        count = len(lane_results[0].boxes)
                        self.log_detection(f"🚗 Lane: {count} detected")
                        
                        # Draw lane detections on display frame
                        for box in lane_results[0].boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            conf = float(box.conf[0])
                            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(display_frame, f"Lane {conf:.2f}", (x1, y1-10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                except Exception as e:
                    print(f"Lane detection error: {e}")
                    import traceback
                    traceback.print_exc()
            elif self.show_lane_detections:
                # Model not available but toggle is on
                if not hasattr(self.hsv, 'lane_model'):
                    print("Lane model attribute doesn't exist on HSV instance")
                elif self.hsv.lane_model is None:
                    print("Lane model is None - model not loaded")
            
            # Run barrel detection
            if self.show_barrel_detections and hasattr(self.hsv, 'barrel_model') and self.hsv.barrel_model is not None:
                try:
                    barrel_results = self.hsv.barrel_model(frame, conf=0.5, verbose=False)
                    if len(barrel_results) > 0 and len(barrel_results[0].boxes) > 0:
                        count = len(barrel_results[0].boxes)
                        self.log_detection(f"🛢️ Barrel: {count} detected")
                        
                        # Draw barrel detections on display frame
                        for box in barrel_results[0].boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            conf = float(box.conf[0])
                            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                            cv2.putText(display_frame, f"Barrel {conf:.2f}", (x1, y1-10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                except Exception as e:
                    print(f"Barrel detection error: {e}")
        
        # Display both video (with YOLO overlays) and mask
        self.display_image(display_frame, self.label_video)
        
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
        app.exec()