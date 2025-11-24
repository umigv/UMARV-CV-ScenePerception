"""Qt-based multi-view mask viewer for post-processing with timeline and bookmarks"""
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QGridLayout, QPushButton,
                             QTextEdit, QSplitter, QCheckBox, QSlider, QGroupBox,
                             QSizePolicy)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap, QTextCursor, QFont
import cv2
import numpy as np
import sys
import platform
from datetime import datetime


class ConsoleWidget(QTextEdit):
    """Custom console widget for displaying YOLO output"""
    
    def __init__(self):
        super().__init__()
        self.setReadOnly(True)
        self.setMaximumHeight(200)
        
        # Styling
        font = QFont("Courier New", 9)
        self.setFont(font)
        self.setStyleSheet("""
            QTextEdit {
                background-color: #0a0a0a;
                color: #00ff00;
                border: 2px solid #333;
                border-radius: 5px;
                padding: 5px;
            }
        """)
        
        # Limits
        self.max_lines = 100  # Keep last 100 lines
        
    def append_text(self, text):
        """Append text and auto-scroll"""
        self.moveCursor(QTextCursor.MoveOperation.End)
        self.insertPlainText(text)
        self.moveCursor(QTextCursor.MoveOperation.End)
        
        # Limit lines
        text_content = self.toPlainText()
        lines = text_content.split('\n')
        if len(lines) > self.max_lines:
            # Keep only last max_lines
            self.clear()
            self.setPlainText('\n'.join(lines[-self.max_lines:]))
            self.moveCursor(QTextCursor.MoveOperation.End)
    
    def log(self, message, level="INFO"):
        """Log a message with timestamp"""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        formatted = f"[{timestamp}] {level}: {message}\n"
        self.append_text(formatted)


class MaskViewerWindow(QMainWindow):
    """Qt window to display all masks with timeline and bookmarks"""
    
    def __init__(self, hsv_instance, image_mode=False):
        super().__init__()
        self.hsv = hsv_instance
        self.image_mode = image_mode
        self.cap = None
        self.timer = QTimer()
        
        # Pause state
        self.paused = False
        
        # Performance settings
        self.frame_skip = 1
        self.frame_counter = 0
        self.displayed_frame_counter = 0  # Track actual displayed frames for FPS
        
        # Video timeline tracking
        self.total_frames = 0
        self.current_frame_pos = 0
        self.video_fps = 30
        self.is_seeking = False
        
        # Landmark/bookmark system
        self.landmarks = []
        self.active_landmark = None
        self.loop_start_frame = None
        self.loop_end_frame = None
        
        # Detection tracking
        self.last_detections = {}
        
        # Frame storage
        self.current_frame = None
        self.last_frame = None
        
        self.setup_ui()
        if not image_mode:
            self.timer.timeout.connect(self.update_frame)
        
    def setup_ui(self):
        self.setWindowTitle("ARV HSV Viewer - All Masks")
        
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
            self.setAttribute(Qt.WidgetAttribute.WA_NativeWindow, True)
        
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        
        # Left: Landmarks panel (only for video mode)
        if not self.image_mode:
            left_panel = QWidget()
            left_layout = QVBoxLayout(left_panel)
            left_layout.setContentsMargins(0, 0, 0, 0)
            left_layout.setSpacing(10)
            
            # HSV Values panel at top
            hsv_panel = self.create_hsv_values_panel()
            left_layout.addWidget(hsv_panel)
            
            # Landmarks panel below
            landmarks_panel = self.create_landmarks_panel()
            left_layout.addWidget(landmarks_panel)
            
            main_layout.addWidget(left_panel)
        
        # Middle/Right: Video content
        content_layout = QVBoxLayout()
        
        # Title
        title = QLabel("<h2>Multi Mask Viewer</h2>")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("color: white; padding: 10px;")
        content_layout.addWidget(title)
        
        # Splitter for video grid and console
        splitter = QSplitter(Qt.Orientation.Vertical)
        
        # Top: Video grid
        video_widget = QWidget()
        video_layout = QVBoxLayout(video_widget)
        
        self.grid_layout = QGridLayout()
        self.grid_layout.setSpacing(10)
        
        # Create labels for display
        self.video_labels = {}
        
        # Row 0: Original and Combined
        self.video_labels['original'] = self.create_video_label("Original Frame")
        self.video_labels['combined'] = self.create_video_label("Combined Mask")
        self.grid_layout.addWidget(self.video_labels['original'], 0, 0)
        self.grid_layout.addWidget(self.video_labels['combined'], 0, 1)
        
        # Row 1+: Individual masks (will be added dynamically)
        self.mask_labels = {}
        
        video_layout.addLayout(self.grid_layout)
        splitter.addWidget(video_widget)
        
        # Bottom: Console
        console_widget = QWidget()
        console_layout = QVBoxLayout(console_widget)
        
        console_header = QLabel("<b>📊 Detection Console (YOLO Output)</b>")
        console_header.setStyleSheet("color: #14a085; padding: 5px;")
        console_layout.addWidget(console_header)
        
        self.console = ConsoleWidget()
        console_layout.addWidget(self.console)
        
        splitter.addWidget(console_widget)
        
        # Set initial sizes (60% video, 40% console) - console visible by default
        splitter.setSizes([600, 400])
        
        content_layout.addWidget(splitter)
        
        # Timeline controls (only for video mode)
        if not self.image_mode:
            timeline_group = self.create_timeline_controls()
            content_layout.addWidget(timeline_group)
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        if not self.image_mode:
            self.btn_pause = QPushButton("⏸ Pause")
            self.btn_pause.clicked.connect(self.toggle_pause)
            button_layout.addWidget(self.btn_pause)
        
        btn_clear_console = QPushButton("🗑 Clear Console")
        btn_clear_console.clicked.connect(self.console.clear)
        button_layout.addWidget(btn_clear_console)
        
        self.btn_quit = QPushButton("✕ Close")
        self.btn_quit.clicked.connect(self.close)
        button_layout.addWidget(self.btn_quit)

        # YOLO Controls
        yolo_label = QLabel("YOLO Models:")
        yolo_label.setStyleSheet("color: #888; padding: 0 10px;")
        button_layout.addWidget(yolo_label)
        
        self.checkbox_lane_yolo = QCheckBox("Lane Lines")
        self.checkbox_lane_yolo.setStyleSheet("color: white;")
        self.checkbox_lane_yolo.setChecked(False)
        self.checkbox_lane_yolo.stateChanged.connect(self.toggle_lane_yolo)
        button_layout.addWidget(self.checkbox_lane_yolo)
        
        self.checkbox_barrel_yolo = QCheckBox("Barrels")
        self.checkbox_barrel_yolo.setStyleSheet("color: white;")
        self.checkbox_barrel_yolo.setChecked(False)
        self.checkbox_barrel_yolo.stateChanged.connect(self.toggle_barrel_yolo)
        button_layout.addWidget(self.checkbox_barrel_yolo)
            
        # Stats
        self.stats_label = QLabel("Frame: 0 | Detections: --")
        self.stats_label.setStyleSheet("color: #888;")
        button_layout.addWidget(self.stats_label)
        
        # FPS info
        self.fps_label = QLabel("FPS: --")
        self.fps_label.setStyleSheet("color: #888;")
        button_layout.addWidget(self.fps_label)
        
        button_layout.addStretch()
        content_layout.addLayout(button_layout)
        
        main_layout.addLayout(content_layout)
        
        # Styling
        self.setStyleSheet("""
            QMainWindow { background-color: #1a1a1a; }
            QLabel { color: white; }
            QPushButton {
                background-color: #0d7377;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #14a085; }
            QPushButton:pressed { background-color: #0a5f63; }
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
    
    def create_hsv_values_panel(self):
        """Create HSV values display panel"""
        panel = QGroupBox("HSV Filter Values")
        panel.setMaximumWidth(250)
        panel.setMinimumWidth(250)  # Fixed width
        
        # Prevent panel from resizing parent
        panel.setSizePolicy(
            QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
        )
        
        layout = QVBoxLayout()
        
        # Create display for each filter
        for filter_name, bounds in self.hsv.hsv_filters.items():
            filter_widget = QWidget()
            filter_layout = QVBoxLayout(filter_widget)
            filter_layout.setContentsMargins(5, 5, 5, 5)
            filter_layout.setSpacing(2)
            
            # Filter name
            name_label = QLabel(f"<b>{filter_name.title()}</b>")
            name_label.setStyleSheet("color: #14a085; font-size: 11px;")
            filter_layout.addWidget(name_label)
            
            # HSV values in compact format
            h_label = QLabel(f"H: {bounds['h_lower']}-{bounds['h_upper']}")
            h_label.setStyleSheet("color: #aaa; font-size: 9px; font-family: monospace;")
            filter_layout.addWidget(h_label)
            
            s_label = QLabel(f"S: {bounds['s_lower']}-{bounds['s_upper']}")
            s_label.setStyleSheet("color: #aaa; font-size: 9px; font-family: monospace;")
            filter_layout.addWidget(s_label)
            
            v_label = QLabel(f"V: {bounds['v_lower']}-{bounds['v_upper']}")
            v_label.setStyleSheet("color: #aaa; font-size: 9px; font-family: monospace;")
            filter_layout.addWidget(v_label)
            
            # Separator line
            separator = QWidget()
            separator.setFixedHeight(1)
            separator.setStyleSheet("background-color: #333;")
            filter_layout.addWidget(separator)
            
            layout.addWidget(filter_widget)
        
        layout.addStretch()
        panel.setLayout(layout)
        return panel
    
    def create_landmarks_panel(self):
        """Create landmarks/bookmarks panel"""
        panel = QGroupBox("Landmarks (Press 'T')")
        panel.setMaximumWidth(250)
        panel.setMinimumWidth(250)  # Fixed width
        
        # Prevent panel from resizing parent
        panel.setSizePolicy(
            QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        )
        
        layout = QVBoxLayout()
        
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
        
        # Check if landmark already exists
        for lm in self.landmarks:
            if abs(lm['frame'] - frame_pos) < self.video_fps:
                self.console.log(f"Landmark already exists near {self.format_time(time_sec)}", "WARN")
                return
        
        landmark = {
            'frame': frame_pos,
            'time': time_sec,
            'window': 3
        }
        self.landmarks.append(landmark)
        self.create_landmark_widget(len(self.landmarks) - 1, landmark)
        self.console.log(f"✓ Landmark added at {self.format_time(time_sec)}", "INFO")
    
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
        loop_row.setSpacing(3)
        
        loop_btn = QPushButton("⟲ Loop")
        loop_btn.setCheckable(True)
        loop_btn.setFixedWidth(60)
        loop_btn.setStyleSheet("""
            QPushButton {
                background-color: #4a4a4a;
                color: white;
                border: none;
                padding: 3px;
                font-size: 9px;
            }
            QPushButton:checked {
                background-color: #0d7377;
            }
            QPushButton:hover { background-color: #5a5a5a; }
        """)
        loop_btn.clicked.connect(lambda checked: self.toggle_landmark_loop(index, checked))
        loop_row.addWidget(loop_btn)
        
        # Window size controls
        minus_btn = QPushButton("-")
        minus_btn.setFixedSize(20, 20)
        minus_btn.setStyleSheet("""
            QPushButton {
                background-color: #555;
                color: white;
                border: none;
                font-size: 10px;
                padding: 0;
            }
            QPushButton:hover { background-color: #666; }
        """)
        minus_btn.clicked.connect(lambda: self.adjust_landmark_window(index, -1, window_label))
        loop_row.addWidget(minus_btn)
        
        window_label = QLabel(f"±{landmark['window']}s")
        window_label.setStyleSheet("color: white; font-size: 9px;")
        window_label.setFixedWidth(35)
        window_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        loop_row.addWidget(window_label)
        
        plus_btn = QPushButton("+")
        plus_btn.setFixedSize(20, 20)
        plus_btn.setStyleSheet("""
            QPushButton {
                background-color: #555;
                color: white;
                border: none;
                font-size: 10px;
                padding: 0;
            }
            QPushButton:hover { background-color: #666; }
        """)
        plus_btn.clicked.connect(lambda: self.adjust_landmark_window(index, 1, window_label))
        loop_row.addWidget(plus_btn)
        
        container_layout.addLayout(loop_row)
        
        # Store references
        landmark['widget'] = container
        landmark['loop_btn'] = loop_btn
        landmark['window_label'] = window_label
        
        self.landmarks_layout.insertWidget(len(self.landmarks) - 1, container)
    
    def jump_to_landmark(self, index):
        """Jump to a specific landmark"""
        if index >= len(self.landmarks):
            return
        
        landmark = self.landmarks[index]
        frame_pos = landmark['frame']
        
        # Pause and seek
        was_playing = not self.paused
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
            self.process_and_display_frame(frame)
        
        self.update_time_display()
        self.console.log(f"→ Jumped to landmark at {self.format_time(landmark['time'])}", "INFO")
    
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
            
            self.console.log(f"⟲ Loop: {self.format_time(self.loop_start_frame / self.video_fps)} - {self.format_time(self.loop_end_frame / self.video_fps)}", "INFO")
            
            # Jump to start of loop
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.loop_start_frame)
            self.current_frame_pos = self.loop_start_frame
            
            # Start playing if paused
            if self.paused:
                self.toggle_pause()
        else:
            # Deactivate loop
            self.active_landmark = None
            self.loop_start_frame = None
            self.loop_end_frame = None
            self.console.log("⟲ Loop mode disabled", "INFO")
    
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
        
        # Update indices
        if self.active_landmark is not None and self.active_landmark > index:
            self.active_landmark -= 1
        
        self.console.log(f"✕ Landmark deleted", "INFO")
    
    def create_timeline_controls(self):
        """Create video timeline scrubber and playback controls"""
        timeline_group = QGroupBox("Video Timeline")
        timeline_layout = QVBoxLayout()
        
        # Playback controls row
        controls_layout = QHBoxLayout()
        
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
        self.timeline_slider.sliderMoved.connect(self.on_timeline_moved)
        self.timeline_slider.valueChanged.connect(self.on_timeline_clicked)
        
        timeline_layout.addWidget(self.timeline_slider)
        
        timeline_group.setLayout(timeline_layout)
        return timeline_group
    
    def on_timeline_pressed(self):
        """User started dragging timeline"""
        self.is_seeking = True
        self.timer.stop()
    
    def on_timeline_released(self):
        """User released timeline"""
        self.is_seeking = False
        
        # Update current_frame_pos
        if self.cap is not None:
            self.current_frame_pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        
        # Only resume timer if not paused
        if not self.paused:
            self.timer.start()
    
    def on_timeline_moved(self, value):
        """Timeline slider dragged - seek to position"""
        if self.cap is not None and self.total_frames > 0:
            frame_pos = int((value / 1000.0) * self.total_frames)
            frame_pos = max(0, min(frame_pos, self.total_frames - 1))
            
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
            self.current_frame_pos = frame_pos
            
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame.copy()
                self.process_and_display_frame(frame)
            
            self.update_time_display()
    
    def on_timeline_clicked(self, value):
        """Timeline slider clicked (not dragged) - seek to position"""
        if not self.is_seeking and self.cap is not None and self.total_frames > 0:
            was_paused = self.paused
            self.timer.stop()
            
            frame_pos = int((value / 1000.0) * self.total_frames)
            frame_pos = max(0, min(frame_pos, self.total_frames - 1))
            
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
            self.current_frame_pos = frame_pos
            
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame.copy()
                self.process_and_display_frame(frame)
            
            self.update_time_display()
            
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

    def toggle_lane_yolo(self, state):
        """Toggle lane line YOLO model"""
        # PySide6 stateChanged emits int (0=unchecked, 2=checked)
        enabled = bool(state)  # 0->False, 2->True
        self.hsv.set_yolo_usage(lane=enabled, barrel=self.hsv.use_barrel_yolo)
        # Check the actual state AFTER setting it
        actual_state = self.hsv.use_lane_yolo
        self.console.log(f"Lane YOLO: {'ENABLED' if actual_state else 'DISABLED'}", "INFO")
        
        # Refresh display immediately
        if self.image_mode:
            self.display_static_image()
        elif self.last_frame is not None:
            self.process_and_display_frame(self.last_frame)
    
    def toggle_barrel_yolo(self, state):
        """Toggle barrel YOLO model"""
        # PySide6 stateChanged emits int (0=unchecked, 2=checked)
        enabled = bool(state)  # 0->False, 2->True
        self.hsv.set_yolo_usage(lane=self.hsv.use_lane_yolo, barrel=enabled)
        # Check the actual state AFTER setting it
        actual_state = self.hsv.use_barrel_yolo
        self.console.log(f"Barrel YOLO: {'ENABLED' if actual_state else 'DISABLED'}", "INFO")
        
        # Refresh display immediately
        if self.image_mode:
            self.display_static_image()
        elif self.last_frame is not None:
            self.process_and_display_frame(self.last_frame)
    
    def create_video_label(self, title):
        """Create a labeled video display widget"""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        
        title_label = QLabel(f"<b>{title}</b>")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("color: #14a085; padding: 5px;")
        layout.addWidget(title_label)
        
        video_label = QLabel()
        video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        video_label.setMinimumSize(640, 360)
        video_label.setStyleSheet(
            "background-color: #0a0a0a; "
            "border: 2px solid #333; "
            "border-radius: 5px;"
        )
        video_label.setScaledContents(False)
        layout.addWidget(video_label)
        
        container.display_label = video_label
        return container
    
    def setup_mask_labels(self, mask_names):
        """Dynamically create labels for individual masks"""
        for label in self.mask_labels.values():
            self.grid_layout.removeWidget(label)
            label.deleteLater()
        self.mask_labels.clear()
        
        row = 1
        col = 0
        for mask_name in mask_names:
            label = self.create_video_label(f"{mask_name.title()} Mask")
            self.mask_labels[mask_name] = label
            self.grid_layout.addWidget(label, row, col)
            
            col += 1
            if col >= 2:
                col = 0
                row += 1
    
    def start(self):
        """Initialize video/image capture and start playback"""
        # Initialize YOLO states if not set (they should already be False from __init__)
        if not hasattr(self.hsv, 'use_lane_yolo'):
            self.hsv.use_lane_yolo = False
        if not hasattr(self.hsv, 'use_barrel_yolo'):
            self.hsv.use_barrel_yolo = False
        
        if self.image_mode:
            frame = cv2.imread(self.hsv.video_path)
            if frame is None:
                self.console.log(f"Unable to open image: {self.hsv.video_path}", "ERROR")
                return False
            
            self.current_frame = frame
            self.console.log(f"Image loaded: {self.hsv.video_path}", "INFO")
            self.console.log(f"Image size: {frame.shape[1]}x{frame.shape[0]}", "INFO")
            self.console.log(f"Active filters: {list(self.hsv.hsv_filters.keys())}", "INFO")
            
            self.setup_mask_labels(list(self.hsv.hsv_filters.keys()))
            self.display_static_image()
            
            self.paused = False
            return True
        else:
            self.cap = cv2.VideoCapture(self.hsv.video_path)
            if not self.cap.isOpened():
                self.console.log(f"Unable to open video: {self.hsv.video_path}", "ERROR")
                return False
            
            self.video_fps = self.cap.get(cv2.CAP_PROP_FPS)
            if self.video_fps <= 0 or self.video_fps > 240:
                self.video_fps = 30
            
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Initialize timeline slider
            if hasattr(self, 'timeline_slider'):
                self.timeline_slider.setMaximum(1000)
            
            self.console.log(f"Video loaded: {self.hsv.video_path}", "INFO")
            self.console.log(f"Native FPS: {self.video_fps:.1f}", "INFO")
            self.console.log(f"Total frames: {self.total_frames}", "INFO")
            if self.total_frames > 0:
                duration = self.total_frames / self.video_fps
                self.console.log(f"Duration: {self.format_time(duration)}", "INFO")
            self.console.log(f"Active filters: {list(self.hsv.hsv_filters.keys())}", "INFO")
            self.console.log("Press 'T' to bookmark frames | Space to pause", "INFO")
            
            self.setup_mask_labels(list(self.hsv.hsv_filters.keys()))
            
            # Calculate timer interval based on video FPS
            timer_interval = int(1000 / self.video_fps)  # milliseconds per frame
            self.console.log(f"Timer interval: {timer_interval}ms ({self.video_fps} FPS)", "INFO")
            
            self.timer.start(timer_interval)  # Use FPS-based interval instead of 1ms
            self.paused = False
            
            # FPS tracking
            self.frame_times = []
            self.last_fps_time = cv2.getTickCount()
            
            return True
        
    def display_static_image(self):
        """Display static image with all masks"""
        frame = self.current_frame.copy()
        
        try:
            combined_mask, masks = self.hsv.get_mask(frame)
            detections = self.parse_yolo_detections()
            
            if detections:
                det_str = ", ".join([f"{k}: {v}" for k, v in detections.items()])
                self.console.log(f"🎯 Detections: {det_str}", "DETECT")
            
            det_count = sum(detections.values()) if detections else 0
            self.stats_label.setText(f"Image | Detections: {det_count}")
            
        except Exception as e:
            self.console.log(f"Error in get_mask: {e}", "ERROR")
            return
        
        self.display_image(frame, self.video_labels['original'].display_label)
        
        if len(combined_mask.shape) == 2:
            combined_bgr = cv2.cvtColor(combined_mask, cv2.COLOR_GRAY2BGR)
        else:
            combined_bgr = combined_mask
        self.display_image(combined_bgr, self.video_labels['combined'].display_label)
        
        for mask_name, mask in masks.items():
            if mask_name in self.mask_labels:
                if len(mask.shape) == 2:
                    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                else:
                    mask_bgr = mask
                self.display_image(mask_bgr, self.mask_labels[mask_name].display_label)
    
    def toggle_pause(self):
        """Toggle pause/play"""
        if self.image_mode:
            self.console.log("Pause/Play not available in image mode", "INFO")
            return
        
        if self.paused:
            self.timer.start()
            self.btn_pause.setText("⏸ Pause")
            self.paused = False
            self.console.log("Playback resumed", "INFO")
        else:
            self.timer.stop()
            self.btn_pause.setText("▶ Play")
            self.paused = True
            self.console.log("Playback paused", "INFO")
    
    def parse_yolo_detections(self):
        """Extract detection info from YOLO models"""
        detections = {}
        
        if self.hsv.use_barrel_yolo:
            if hasattr(self.hsv, 'barrel_boxes') and self.hsv.barrel_boxes is not None:
                barrel_count = len(self.hsv.barrel_boxes)
                if barrel_count > 0:
                    detections['barrels'] = barrel_count
        
        if self.hsv.use_lane_yolo:
            if hasattr(self.hsv, 'lane_masks') and self.hsv.lane_masks is not None:
                lane_count = len(self.hsv.lane_masks)
                if lane_count > 0:
                    detections['lane_lines'] = lane_count
        
        return detections
    
    def update_frame(self):
        """Process and display video frame with frame skipping"""
        # Don't update if seeking or paused
        if self.is_seeking or self.paused:
            return
        
        self.frame_counter += 1
        
        if self.frame_skip > 1 and self.frame_counter % self.frame_skip != 0:
            ret, _ = self.cap.read()
            if not ret:
                # Check if we're in loop mode
                if self.loop_start_frame is not None:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.loop_start_frame)
                    self.current_frame_pos = self.loop_start_frame
                else:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.frame_counter = 0
                    self.current_frame_pos = 0
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
                self.console.log("Video looped", "INFO")
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
        
        # Update timeline slider
        if hasattr(self, 'timeline_slider') and self.total_frames > 0:
            timeline_value = int((self.current_frame_pos / self.total_frames) * 1000)
            self.timeline_slider.blockSignals(True)
            self.timeline_slider.setValue(timeline_value)
            self.timeline_slider.blockSignals(False)
            self.update_time_display()
        
        self.current_frame = frame.copy()
        self.last_frame = frame.copy()
        self.displayed_frame_counter += 1  # Count actual displayed frames
        self.process_and_display_frame(frame)
        
    def process_and_display_frame(self, frame): 
        """Process frame and update all displays"""
        try:
            combined_mask, masks = self.hsv.get_mask(frame)
            detections = self.parse_yolo_detections()
            
            if detections != self.last_detections:
                if detections:
                    det_str = ", ".join([f"{k}: {v}" for k, v in detections.items()])
                    yolo_status = []
                    if self.hsv.use_lane_yolo:
                        yolo_status.append("Lane✓")
                    if self.hsv.use_barrel_yolo:
                        yolo_status.append("Barrel✓")
                    status = f" [{', '.join(yolo_status)}]" if yolo_status else ""
                    self.console.log(f"🎯 Detections: {det_str}{status}", "DETECT")
                
                self.last_detections = detections.copy()
            
            det_count = sum(detections.values()) if detections else 0
            self.stats_label.setText(f"Frame: {self.current_frame_pos} | Detections: {det_count}")
            
        except Exception as e:
            self.console.log(f"Error in get_mask: {e}", "ERROR")
            return
        
        # Create display frame with YOLO overlays
        display_frame = frame.copy()
        
        # Draw YOLO bounding boxes if enabled
        if self.hsv.use_lane_yolo and hasattr(self.hsv, 'lane_model') and self.hsv.lane_model is not None:
            try:
                lane_results = self.hsv.lane_model(frame, conf=0.5, verbose=False)
                if len(lane_results) > 0 and len(lane_results[0].boxes) > 0:
                    for box in lane_results[0].boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(display_frame, f"Lane {conf:.2f}", (x1, y1-10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            except Exception:
                pass
        
        if self.hsv.use_barrel_yolo and hasattr(self.hsv, 'barrel_model') and self.hsv.barrel_model is not None:
            try:
                barrel_results = self.hsv.barrel_model(frame, conf=0.5, verbose=False)
                if len(barrel_results) > 0 and len(barrel_results[0].boxes) > 0:
                    for box in barrel_results[0].boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        cv2.putText(display_frame, f"Barrel {conf:.2f}", (x1, y1-10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            except Exception:
                pass
        
        # Display frame with overlays
        self.display_image(display_frame, self.video_labels['original'].display_label)
        
        if len(combined_mask.shape) == 2:
            combined_bgr = cv2.cvtColor(combined_mask, cv2.COLOR_GRAY2BGR)
        else:
            combined_bgr = combined_mask
        self.display_image(combined_bgr, self.video_labels['combined'].display_label)
        
        for mask_name, mask in masks.items():
            if mask_name in self.mask_labels:
                if len(mask.shape) == 2:
                    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                else:
                    mask_bgr = mask
                self.display_image(mask_bgr, self.mask_labels[mask_name].display_label)
        
        if not self.image_mode:
            self.update_fps()
    
    def display_image(self, img, label):
        """Convert OpenCV image to Qt and display"""
        if img is None:
            return
        
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        display_width = 640
        display_height = 360
        
        img = cv2.resize(img, (display_width, display_height), 
                        interpolation=cv2.INTER_LINEAR)
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = img_rgb.shape
        
        img_rgb = np.ascontiguousarray(img_rgb)
        
        bytes_per_line = ch * w
        qt_image = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        label.setPixmap(pixmap)
    
    def update_fps(self):
        """Calculate and display FPS"""
        current_time = cv2.getTickCount()
        time_diff = (current_time - self.last_fps_time) / cv2.getTickFrequency()
        
        if time_diff >= 1.0:  # Update every second
            fps = self.displayed_frame_counter / time_diff
            self.fps_label.setText(f"FPS: {fps:.1f}")
            
            if fps >= 25:
                self.fps_label.setStyleSheet("color: #51cf66;")
            elif fps >= 15:
                self.fps_label.setStyleSheet("color: #ffd43b;")
            else:
                self.fps_label.setStyleSheet("color: #ff6b6b;")
            
            self.displayed_frame_counter = 0  # Reset displayed frame counter
            self.last_fps_time = current_time
    
    def closeEvent(self, event):
        """Cleanup when window closes"""
        self.console.log("Closing viewer...", "INFO")
        self.timer.stop()
        if self.cap:
            self.cap.release()
        event.accept()


def view_all_masks(hsv_instance, image_mode=False):
    """Launch Qt viewer with timeline and bookmarks"""
    import os
    
    if platform.system() == "Darwin":
        os.environ["QT_MAC_WANTS_LAYER"] = "1"
        os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "0"
    
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
        
        if platform.system() == "Darwin":
            app.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, False)
    
    window = MaskViewerWindow(hsv_instance, image_mode=image_mode)
    if window.start():
        window.show()
        
        mode_text = "Image" if image_mode else "Video"
        print("\n" + "="*60)
        print(f"Multi-Mask Viewer Started ({mode_text} Mode)")
        print("="*60)
        print("Features:")
        print("  • Multi-view display (original + all masks)")
        print("  • Live detection console (YOLO output)")
        print("  • Timeline scrubber with bookmarks")
        print("  • Landmark system (Press 'T' to bookmark)")
        print("  • Loop mode for focused tuning")
        print("\nKeyboard Shortcuts:")
        print("  • T - Add bookmark at current frame")
        print("  • Space - Pause/Play")
        print("\nDisplaying:")
        print("  • Original Frame")
        print("  • Combined Mask (all detections)")
        for mask_name in hsv_instance.hsv_filters.keys():
            print(f"  • {mask_name.title()} Mask")
        print("="*60 + "\n")
        
        app.exec()