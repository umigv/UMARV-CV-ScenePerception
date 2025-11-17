"""Qt-based multi-view mask viewer for post-processing with live console"""
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QGridLayout, QPushButton,
                             QTextEdit, QSplitter,QCheckBox)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QTextCursor, QFont
import cv2
import numpy as np
import sys
import platform
from datetime import datetime
import io
from contextlib import redirect_stdout


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
        self.moveCursor(QTextCursor.End)
        self.insertPlainText(text)
        self.moveCursor(QTextCursor.End)
        
        # Limit lines
        text_content = self.toPlainText()
        lines = text_content.split('\n')
        if len(lines) > self.max_lines:
            # Keep only last max_lines
            self.clear()
            self.setPlainText('\n'.join(lines[-self.max_lines:]))
            self.moveCursor(QTextCursor.End)
    
    def log(self, message, level="INFO"):
        """Log a message with timestamp"""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        formatted = f"[{timestamp}] {level}: {message}\n"
        self.append_text(formatted)


class MaskViewerWindow(QMainWindow):
    """Qt window to display all masks in a grid layout with live console"""
    
    def __init__(self, hsv_instance, image_mode=False):
        super().__init__()
        self.hsv = hsv_instance
        self.image_mode = image_mode  # NEW
        self.cap = None
        self.timer = QTimer()
        
        # Performance settings
        self.frame_skip = 1
        self.frame_counter = 0
        
        # Detection tracking
        self.last_detections = {}
        
        self.setup_ui()
        if not image_mode:
            self.timer.timeout.connect(self.update_frame)
        
    def setup_ui(self):
        self.setWindowTitle("ARV HSV Viewer - All Masks")
        self.setGeometry(50, 50, 1600, 1200)
        
        # M1 optimization
        if platform.system() == "Darwin":
            self.setAttribute(Qt.WA_NativeWindow, True)
        
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        
        # Title
        title = QLabel("<h2>Multi-Mask Viewer with Live Detection Console</h2>")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: white; padding: 10px;")
        main_layout.addWidget(title)
        
        # Splitter for video grid and console
        splitter = QSplitter(Qt.Vertical)
        
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
        
        # Set initial sizes (80% video, 20% console)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 1)
        
        main_layout.addWidget(splitter)
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        self.btn_pause = QPushButton("⏸ Pause")
        self.btn_pause.clicked.connect(self.toggle_pause)
        button_layout.addWidget(self.btn_pause)
        
        btn_clear_console = QPushButton("🗑 Clear Console")
        btn_clear_console.clicked.connect(self.console.clear)
        button_layout.addWidget(btn_clear_console)
        
        self.btn_quit = QPushButton("✕ Close")
        self.btn_quit.clicked.connect(self.close)
        button_layout.addWidget(self.btn_quit)

        # NEW: YOLO Controls
    
        yolo_label = QLabel("YOLO Models:")
        yolo_label.setStyleSheet("color: #888; padding: 0 10px;")
        button_layout.addWidget(yolo_label)
        
        self.checkbox_lane_yolo = QCheckBox("Lane Lines")
        self.checkbox_lane_yolo.setStyleSheet("color: white;")
        self.checkbox_lane_yolo.setChecked(False)  # Start disabled
        self.checkbox_lane_yolo.stateChanged.connect(self.toggle_lane_yolo)
        button_layout.addWidget(self.checkbox_lane_yolo)
        
        self.checkbox_barrel_yolo = QCheckBox("Barrels")
        self.checkbox_barrel_yolo.setStyleSheet("color: white;")
        self.checkbox_barrel_yolo.setChecked(False)  # Start disabled
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
        main_layout.addLayout(button_layout)
        
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
        """)

    def toggle_lane_yolo(self, state):
        """Toggle lane line YOLO model"""
        enabled = (state == Qt.Checked)
        self.hsv.set_yolo_usage(lane=enabled, barrel=self.hsv.use_barrel_yolo)
        self.console.log(f"Lane YOLO: {'ENABLED' if enabled else 'DISABLED'}", "INFO")
        
        # Refresh display in image mode
        if self.image_mode:
            self.display_static_image()
    
    def toggle_barrel_yolo(self, state):
        """Toggle barrel YOLO model"""
        enabled = (state == Qt.Checked)
        self.hsv.set_yolo_usage(lane=self.hsv.use_lane_yolo, barrel=enabled)
        self.console.log(f"Barrel YOLO: {'ENABLED' if enabled else 'DISABLED'}", "INFO")
        
        # Refresh display in image mode
        if self.image_mode:
            self.display_static_image()
    
    def create_video_label(self, title):
        """Create a labeled video display widget"""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Title label
        title_label = QLabel(f"<b>{title}</b>")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("color: #14a085; padding: 5px;")
        layout.addWidget(title_label)
        
        # Video display label
        video_label = QLabel()
        video_label.setAlignment(Qt.AlignCenter)
        video_label.setMinimumSize(640, 360)
        video_label.setStyleSheet(
            "background-color: #0a0a0a; "
            "border: 2px solid #333; "
            "border-radius: 5px;"
        )
        video_label.setScaledContents(False)
        layout.addWidget(video_label)
        
        # Store reference to the actual display label
        container.display_label = video_label
        return container
    
    def setup_mask_labels(self, mask_names):
        """Dynamically create labels for individual masks"""
        # Clear existing mask labels
        for label in self.mask_labels.values():
            self.grid_layout.removeWidget(label)
            label.deleteLater()
        self.mask_labels.clear()
        
        # Create new labels for each mask
        row = 1
        col = 0
        for mask_name in mask_names:
            label = self.create_video_label(f"{mask_name.title()} Mask")
            self.mask_labels[mask_name] = label
            self.grid_layout.addWidget(label, row, col)
            
            col += 1
            if col >= 2:  # 2 columns
                col = 0
                row += 1
    
    def start(self):
        """Initialize video/image capture and start playback"""
        if self.image_mode:
            # Load single image
            frame = cv2.imread(self.hsv.video_path)
            if frame is None:
                self.console.log(f"Unable to open image: {self.hsv.video_path}", "ERROR")
                return False
            
            self.current_frame = frame
            self.console.log(f"Image loaded: {self.hsv.video_path}", "INFO")
            self.console.log(f"Image size: {frame.shape[1]}x{frame.shape[0]}", "INFO")
            self.console.log(f"Active filters: {list(self.hsv.hsv_filters.keys())}", "INFO")
            self.console.log("YOLO models: DISABLED (toggle via checkboxes)", "INFO")
            
            # Setup mask labels
            self.setup_mask_labels(list(self.hsv.hsv_filters.keys()))
            
            # Display static image
            self.display_static_image()
            
            self.paused = False
            return True
        else:
            # Video mode (existing code)
            self.cap = cv2.VideoCapture(self.hsv.video_path)
            if not self.cap.isOpened():
                self.console.log(f"Unable to open video: {self.hsv.video_path}", "ERROR")
                return False
            
            native_fps = self.cap.get(cv2.CAP_PROP_FPS)
            if native_fps <= 0 or native_fps > 240:
                native_fps = 30
            
            self.console.log(f"Video loaded: {self.hsv.video_path}", "INFO")
            self.console.log(f"Native FPS: {native_fps:.1f}", "INFO")
            self.console.log(f"Active filters: {list(self.hsv.hsv_filters.keys())}", "INFO")
            self.console.log("YOLO models: DISABLED (toggle via checkboxes)", "INFO")
            self.console.log("Starting playback...", "INFO")
            
            # Setup mask labels based on available filters
            self.setup_mask_labels(list(self.hsv.hsv_filters.keys()))
            
            # Start timer (1ms for maximum throughput)
            self.timer.start(1)
            self.paused = False
            
            # FPS tracking
            self.frame_times = []
            self.last_fps_time = cv2.getTickCount()
            self.total_frames = 0
            
            return True
        
    def display_static_image(self):
        """Display static image with all masks"""
        frame = self.current_frame.copy()
        
        # Get all masks
        try:
            combined_mask, masks = self.hsv.get_mask(frame)
            
            # Extract detection info
            detections = self.parse_yolo_detections()
            
            # Log detections
            if detections:
                det_str = ", ".join([f"{k}: {v}" for k, v in detections.items()])
                self.console.log(f"🎯 Detections: {det_str}", "DETECT")
            else:
                self.console.log("No detections", "DETECT")
            
            # Update stats
            det_count = sum(detections.values()) if detections else 0
            self.stats_label.setText(f"Image | Detections: {det_count}")
            
        except Exception as e:
            self.console.log(f"Error in get_mask: {e}", "ERROR")
            return
        
        # Display original frame
        self.display_image(frame, self.video_labels['original'].display_label)
        
        # Display combined mask
        if len(combined_mask.shape) == 2:
            combined_bgr = cv2.cvtColor(combined_mask, cv2.COLOR_GRAY2BGR)
        else:
            combined_bgr = combined_mask
        self.display_image(combined_bgr, self.video_labels['combined'].display_label)
        
        # Display individual masks
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
        
        # Get barrel detections (only if enabled)
        if self.hsv.use_barrel_yolo:
            if hasattr(self.hsv, 'barrel_boxes') and self.hsv.barrel_boxes is not None:
                barrel_count = len(self.hsv.barrel_boxes)
                if barrel_count > 0:
                    detections['barrels'] = barrel_count
        
        # Get lane detections (only if enabled)
        # Get lane line detections (only if enabled)
        if self.hsv.use_lane_yolo:
            if hasattr(self.hsv, 'lane_masks') and self.hsv.lane_masks is not None:
                lane_count = len(self.hsv.lane_masks)
                if lane_count > 0:
                    detections['lane_lines'] = lane_count
        
        return detections
    
    def update_frame(self):
        """Process and display video frame with all masks"""
        # Frame skipping for performance
        self.frame_counter += 1
        if self.frame_counter % self.frame_skip != 0:
            ret, _ = self.cap.read()
            if not ret:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.frame_counter = 0
            return
        
        ret, frame = self.cap.read()
        if not ret:
            # Loop video
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.frame_counter = 0
            self.console.log("Video looped", "INFO")
            ret, frame = self.cap.read()
            if not ret:
                return
        
        self.total_frames += 1
        
        # Get all masks (your existing get_mask method)
        # This internally calls update_mask which runs YOLO
        try:
            combined_mask, masks = self.hsv.get_mask(frame)
            
            # Extract detection info
            detections = self.parse_yolo_detections()
            
            # Log detection changes with YOLO status
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
                else:
                    self.console.log("No detections", "DETECT")
                
                self.last_detections = detections.copy()
            
            # Update stats
            det_count = sum(detections.values()) if detections else 0
            self.stats_label.setText(f"Frame: {self.total_frames} | Detections: {det_count}")
            
        except Exception as e:
            self.console.log(f"Error in get_mask: {e}", "ERROR")
            return
        
        # Display original frame
        self.display_image(frame, self.video_labels['original'].display_label)
        
        # Display combined mask (convert to BGR for display)
        if len(combined_mask.shape) == 2:
            combined_bgr = cv2.cvtColor(combined_mask, cv2.COLOR_GRAY2BGR)
        else:
            combined_bgr = combined_mask
        self.display_image(combined_bgr, self.video_labels['combined'].display_label)
        
        # Display individual masks
        for mask_name, mask in masks.items():
            if mask_name in self.mask_labels:
                if len(mask.shape) == 2:
                    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                else:
                    mask_bgr = mask
                self.display_image(mask_bgr, self.mask_labels[mask_name].display_label)
        
        # Update FPS counter
        self.update_fps()
    
    def display_image(self, img, label):
        """Convert OpenCV image to Qt and display - OPTIMIZED"""
        if img is None:
            return
        
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        # Fixed display size for consistent performance
        display_width = 640
        display_height = 360
        
        # Resize with OpenCV (faster than Qt)
        img = cv2.resize(img, (display_width, display_height), 
                        interpolation=cv2.INTER_LINEAR)
        
        # Convert to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = img_rgb.shape
        
        # Ensure contiguous
        img_rgb = np.ascontiguousarray(img_rgb)
        
        # Create QImage and display
        bytes_per_line = ch * w
        qt_image = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        label.setPixmap(pixmap)
    
    def update_fps(self):
        """Calculate and display FPS"""
        current_time = cv2.getTickCount()
        time_diff = (current_time - self.last_fps_time) / cv2.getTickFrequency()
        
        if time_diff >= 1.0:  # Update every second
            fps = self.frame_counter / time_diff
            self.fps_label.setText(f"FPS: {fps:.1f}")
            
            # Color code based on performance
            if fps >= 25:
                self.fps_label.setStyleSheet("color: #51cf66;")  # Green
            elif fps >= 15:
                self.fps_label.setStyleSheet("color: #ffd43b;")  # Yellow
            else:
                self.fps_label.setStyleSheet("color: #ff6b6b;")  # Red
            
            self.frame_counter = 0
            self.last_fps_time = current_time
    
    def closeEvent(self, event):
        """Cleanup when window closes"""
        self.console.log("Closing viewer...", "INFO")
        self.timer.stop()
        if self.cap:
            self.cap.release()
        event.accept()


def view_all_masks(hsv_instance, image_mode=False):
    """
    Launch Qt viewer to display all masks in real-time with detection console
    
    Shows:
    - Original video frame
    - Combined mask (all colors + YOLO)
    - Individual color masks (yellow, white, etc.)
    - Live detection console with YOLO output
    
    Args:
        hsv_instance: Instance of hsv class with tuned filters
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
    
    window = MaskViewerWindow(hsv_instance, image_mode=image_mode)
    if window.start():
        window.show()
        
        mode_text = "Image" if image_mode else "Video"
        print("\n" + "="*60)
        print(f"Multi-Mask Viewer Started ({mode_text} Mode)")
        print("="*60)
        print("Features:")
        print("  • Multi-view video display (original + all masks)")
        print("  • Live detection console (YOLO output)")
        print("  • Real-time FPS monitoring")
        print("  • Scrollable detection history")
        print("\nDisplaying:")
        print("  • Original Frame")
        print("  • Combined Mask (all detections)")
        for mask_name in hsv_instance.hsv_filters.keys():
            print(f"  • {mask_name.title()} Mask")
        print("\nControls:")
        print("  • Pause/Play - Freeze/resume video")
        print("  • Clear Console - Reset detection log")
        print("  • Close - Exit viewer")
        print("="*60 + "\n")
        
        app.exec_()
