from PyQt6.QtWidgets import (
    QLabel, QPushButton, QFileDialog, QWidget, QHBoxLayout, QFrame,
    QFormLayout, QSpinBox, QLineEdit, QSizePolicy, QCheckBox, QComboBox,
    QMessageBox, QVBoxLayout, QGridLayout
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPixmap, QImage

from datetime import datetime
import os
import RPi.GPIO as GPIO
import numpy as np
import random  # For placeholder model data

# Matplotlib for the graph
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from widget_collapsible_box import CollapsibleBox
from config import save_config_atomic
from camera_thread import CameraPreviewThread
from picamera2 import Picamera2
from libcamera import Transform


class CameraWidget(QWidget):
    """
    Camera Control widget using Picamera2.
    - Dual-stream: lores for preview, main for full-res capture
    - Supports JPEG and DNG (raw) capture
    - Channel filtering (RGB, Red, Green, Blue, Grayscale)
    - Persists default camera settings into config.json via save_config_atomic()
    - [NEW] Live Recording with Placeholder Model Analysis
    """

    def __init__(self, left_container: QWidget, left_content_layout: QGridLayout, parent_layout, config):
        super().__init__()
        self.left_container = left_container
        self.left_container_layout = left_content_layout
        self.parent_layout = parent_layout
        self.config = config
        self.config_path = getattr(config, "cfg_path", None)

        self.CAMERA = getattr(self.config, "camera", {})
        self._live_running = False

        # Recording State
        self.is_recording = False
        self.recording_timer = QTimer()
        self.recording_timer.setInterval(10000)  # 10 seconds (in ms)
        self.recording_timer.timeout.connect(self._recording_step)
        self.recording_count = 0
        
        # Data for Graph
        self.graph_data_counts = []
        self.graph_data_time = []

        # LED setup
        self.led_pin = 20
        self.led_on = False
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.led_pin, GPIO.OUT, initial=GPIO.LOW)

        # Default save folder
        self.default_folder = os.path.expanduser("~/Desktop/Users/")
        os.makedirs(self.default_folder, exist_ok=True)
        self.selected_directory = self.default_folder

        self._build_camera_UI()
        
        # Initialize the expanded left panel (Analysis + Graph)
        self._init_left_panel_extras()
        
        self._init_embedded_camera()

    # ---------------- UI ----------------

    def _build_camera_UI(self):
        box = CollapsibleBox("Camera Control")

        panel = QFrame()
        panel_layout = QHBoxLayout(panel)
        panel_layout.setContentsMargins(30, 10, 30, 10)
        panel_layout.setSpacing(8)

        left_coln_layout = QFormLayout()
        left_coln_layout.setContentsMargins(0, 0, 0, 0)

        right_coln_layout = QFormLayout()
        right_coln_layout.setContentsMargins(0, 0, 0, 0)

        # ---------- Left column (settings) ----------
        self.camera_status_label = QLabel("Idle")

        self.camera_width = QSpinBox()
        self.camera_width.setMaximum(10000)
        self.camera_height = QSpinBox()
        self.camera_height.setMaximum(10000)

        self.camera_shutter = QLineEdit()
        self.camera_shutter.setPlaceholderText("e.g., 10000")
        self.camera_contrast = QLineEdit()
        self.camera_contrast.setPlaceholderText("e.g., 1.0")
        self.camera_ev = QLineEdit()
        self.camera_ev.setPlaceholderText("e.g., 0")
        self.camera_prev_time = QLineEdit()
        self.camera_prev_time.setPlaceholderText("e.g., 10000")
        self.camera_gain = QLineEdit()
        self.camera_gain.setPlaceholderText("e.g., 1.5")
        self.camera_digital_gain = QLineEdit()
        self.camera_digital_gain.setPlaceholderText("e.g., 1.0")
        self.camera_saturation = QLineEdit()
        self.camera_saturation.setPlaceholderText("e.g., 1.0")
        self.camera_awbgains = QLineEdit()
        self.camera_awbgains.setPlaceholderText("red,blue")
        self.camera_sharpness = QLineEdit()
        self.camera_sharpness.setPlaceholderText("e.g., 1.0")
        self.camera_brightness = QLineEdit()
        self.camera_brightness.setPlaceholderText("e.g., 0.5")

        # Load defaults from config
        self.camera_width.setValue(int(self._cam_get("default_width", 4056)))
        self.camera_height.setValue(int(self._cam_get("default_height", 3040)))
        self.camera_shutter.setText(str(self._cam_get("default_shutter", "")))
        self.camera_contrast.setText(str(self._cam_get("default_contrast", "")))
        self.camera_gain.setText(str(self._cam_get("default_gain", "")))
        self.camera_awbgains.setText(str(self._cam_get("default_awbgains", "")))
        self.camera_sharpness.setText(str(self._cam_get("default_sharpness", "")))

        left_coln_layout.addRow("Status:", self.camera_status_label)
        left_coln_layout.addRow("Width:", self.camera_width)
        left_coln_layout.addRow("Height:", self.camera_height)
        left_coln_layout.addRow("Exposure (us):", self.camera_shutter)
        left_coln_layout.addRow("Contrast:", self.camera_contrast)
        left_coln_layout.addRow("EV:", self.camera_ev)
        left_coln_layout.addRow("Prev. Time:", self.camera_prev_time)
        left_coln_layout.addRow("AnalogueGain:", self.camera_gain)
        left_coln_layout.addRow("DigitalGain:", self.camera_digital_gain)
        left_coln_layout.addRow("AWB Gains (r,b):", self.camera_awbgains)
        left_coln_layout.addRow("Saturation:", self.camera_saturation)
        left_coln_layout.addRow("Sharpness:", self.camera_sharpness)
        left_coln_layout.addRow("Brightness:", self.camera_brightness)

        # ---------- Right column (actions) ----------
        self.camera_raw_preview = QCheckBox("Raw Preview")
        self.camera_raw_preview.setToolTip("Show RAW Bayer stream as grayscale (fast).")
        self.camera_raw_preview.toggled.connect(self._on_raw_preview_toggled)
        
        self.camera_fullraw = QCheckBox("Save Raw (DNG)")
        self.camera_fullraw.setToolTip("Save RAW Bayer data as DNG alongside JPEG.")

        self.camera_no_prev = QCheckBox("No Preview")
        self.camera_no_prev.setToolTip("Disable the camera preview during capture.")
        
        self.camera_hflip = QCheckBox("Horizontal Flip")
        self.camera_hflip.setToolTip("Flip the image horizontally.")
        
        self.camera_vflip = QCheckBox("Vertical Flip")
        self.camera_vflip.setToolTip("Flip the image vertically.")

        self.camera_denoise = QComboBox()
        self.camera_denoise.setFixedWidth(90)
        self.camera_denoise.addItems(["off", "fast", "high_quality"])
        self.camera_denoise.setToolTip("Select the denoise mode.")

        self.camera_channel = QComboBox()
        self.camera_channel.setFixedWidth(90)
        self.camera_channel.addItems(["RGB", "Red", "Green", "Blue", "Grayscale"])
        self.camera_channel.setToolTip("Display/capture single color channel.")
        self.camera_channel.currentTextChanged.connect(self._on_channel_changed)

        self.camera_quality = QSpinBox()
        self.camera_quality.setToolTip("Set JPEG quality percentage (0-100).")
        self.camera_quality.setRange(0, 100)
        self.camera_quality.setValue(int(self._cam_get("default_quality", 95)))

        self.folder_search_btn = QPushButton("Select Folder to Save")
        self.folder_search_btn.clicked.connect(self.select_save_directory)

        self.selected_directory_label = QLabel(self.default_folder)
        self.selected_directory_label.setWordWrap(True)
        self.selected_directory_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

        self.image_name = QLineEdit()
        self.image_name.setPlaceholderText("Enter image name (e.g., image1)")

        self.capture_btn = QPushButton("Capture && Save Image")
        self.capture_btn.clicked.connect(self.capture_image)

        # --- NEW RECORDING BUTTON ---
        self.record_btn = QPushButton("Start Live Recording (10s)")
        self.record_btn.setCheckable(True)
        self.record_btn.setStyleSheet("QPushButton:checked { background-color: #ffcccc; color: red; }")
        self.record_btn.clicked.connect(self.toggle_recording)

        self.show_live_btn = QPushButton("Show Live")
        self.show_live_btn.clicked.connect(self.toggle_live_embedded)

        self.set_default_btn = QPushButton("Set Default")
        self.set_default_btn.setToolTip("Set the default camera settings for future captures.")
        self.set_default_btn.clicked.connect(self.set_default_camera_settings)

        self.LED_btn = QPushButton("Turn LED On")
        self.LED_btn.clicked.connect(self.toggle_LED)

        # Right layout rows
        box_layout_01 = QHBoxLayout()
        box_layout_01.addWidget(QLabel("Denoise:"))
        box_layout_01.addWidget(self.camera_denoise)
        box_layout_01.addSpacing(10)
        box_layout_01.addWidget(QLabel("Quality:"))
        box_layout_01.addWidget(self.camera_quality)
        box_layout_01.addStretch(1)

        box_layout_02 = QHBoxLayout()
        box_layout_02.addWidget(QLabel("Channel:"))
        box_layout_02.addWidget(self.camera_channel)
        box_layout_02.addStretch(1)

        checkbox_layout_01 = QHBoxLayout()
        checkbox_layout_01.addWidget(self.camera_raw_preview)
        checkbox_layout_01.addWidget(self.camera_no_prev)
        checkbox_layout_01.addWidget(self.camera_fullraw)
        checkbox_layout_01.addStretch(1)

        checkbox_layout_02 = QHBoxLayout()
        checkbox_layout_02.addWidget(self.camera_hflip)
        checkbox_layout_02.addWidget(self.camera_vflip)
        checkbox_layout_02.addStretch(1)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.show_live_btn)
        button_layout.addWidget(self.set_default_btn)
        button_layout.addWidget(self.LED_btn)
        button_layout.addStretch(1)

        right_coln_layout.addRow(box_layout_01)
        right_coln_layout.addRow(box_layout_02)
        right_coln_layout.addRow(checkbox_layout_01)
        right_coln_layout.addRow(checkbox_layout_02)
        right_coln_layout.addRow(self.folder_search_btn)
        right_coln_layout.addRow(self.selected_directory_label)
        right_coln_layout.addRow(QLabel("Image Name:"))
        right_coln_layout.addRow(self.image_name)
        right_coln_layout.addRow(button_layout)
        right_coln_layout.addRow(self.capture_btn)
        right_coln_layout.addRow(self.record_btn) # <--- ADDED TO LAYOUT HERE

        panel_layout.addLayout(left_coln_layout, stretch=1)
        panel_layout.addLayout(right_coln_layout, stretch=1)

        content_layout = QVBoxLayout()
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.addWidget(panel)
        box.setContentLayout(content_layout)

        self.parent_layout.addWidget(box)
        self.camera_group = box

    # ---------------- Left Panel Setup ----------------

    def _init_left_panel_extras(self):
        """
        Configure the Left Panel Grid:
        Row 0: Live Preview (Spans 2 columns)
        Row 1: Model Output (Left) | Detection Graph (Right)
        """
        # 1. Main Live Feed
        self.camera_preview = QLabel()
        self.camera_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera_preview.setStyleSheet("background-color: black;")
        self.camera_preview.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.camera_preview.setMinimumSize(1, 1)

        # 2. Model Output Preview
        self.model_preview = QLabel("Model Output\n(Waiting for Recording)")
        self.model_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.model_preview.setStyleSheet("background-color: #222; color: #888; border: 1px solid #444;")
        self.model_preview.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.model_preview.setScaledContents(True)

        # 3. Detection Graph (Matplotlib)
        self.figure = Figure(figsize=(4, 3), dpi=100)
        # Set dark theme for plot to match app usually, or keep white
        self.figure.patch.set_facecolor('#f0f0f0') 
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title("Detections")
        self.ax.set_xlabel("Time")
        self.ax.set_ylabel("Count")
        self.line, = self.ax.plot([], [], 'r-o') # Initialize empty line

        if self.left_container_layout is not None:
            # Clear existing if any (safety)
            # Add widgets to grid: widget, row, col, rowspan, colspan
            self.left_container_layout.addWidget(self.camera_preview, 0, 0, 1, 2)
            self.left_container_layout.addWidget(self.model_preview, 1, 0)
            self.left_container_layout.addWidget(self.canvas, 1, 1)
            
            # Adjust row stretches: Top (Live) gets more space (2/3), Bottom gets 1/3
            self.left_container_layout.setRowStretch(0, 2)
            self.left_container_layout.setRowStretch(1, 1)


    # ---------------- Helpers ----------------

    def _cam_get(self, key: str, default):
        """Get camera config value (supports dict or attribute access)."""
        cam = self.CAMERA
        if isinstance(cam, dict):
            return cam.get(key, default)
        return getattr(cam, key, default)

    def _set_cam(self, key: str, value):
        """Set camera config value (supports dict or attribute access)."""
        cam = self.CAMERA
        if isinstance(cam, dict):
            cam[key] = value
        else:
            setattr(cam, key, value)

    def _safe_text(self, w: QLineEdit) -> str:
        return w.text().strip()

    # ---------------- Camera Setup ----------------

    def _init_embedded_camera(self):
        if not hasattr(self, "picam2") or self.picam2 is None:
            self.picam2 = Picamera2()

        preview_w, preview_h = (1280, 720)
        capture_w = self.camera_width.value()
        capture_h = self.camera_height.value()

        transform = Transform(
            hflip=1 if self.camera_hflip.isChecked() else 0,
            vflip=1 if self.camera_vflip.isChecked() else 0
        )

        denoise_mode = self._get_denoise_mode()
        preview_controls = {}
        if denoise_mode is not None:
            preview_controls["NoiseReductionMode"] = denoise_mode

        # Dual stream: main=full res for capture, lores=low res for preview
        self.preview_config = self.picam2.create_preview_configuration(
            main={"size": (capture_w, capture_h)},
            lores={"size": (preview_w, preview_h), "format": "RGB888"},
            raw={},
            transform=transform,
            controls=preview_controls
        )
        self.picam2.configure(self.preview_config)

        # Preview thread reads from "lores" stream
        self.preview_thread = CameraPreviewThread(self.picam2, fps=30, stream="lores", parent=self)
        self.preview_thread.frame_ready.connect(self._on_preview_frame)
        self.preview_thread.error.connect(self._on_camera_error)

    def _on_preview_frame(self, qimg):
        pix = QPixmap.fromImage(qimg)
        scaled = pix.scaled(
            self.camera_preview.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.FastTransformation
        )
        self.camera_preview.setPixmap(scaled)

    def _on_camera_error(self, msg: str):
        self.camera_status_label.setText(f"Camera error: {msg}")

    def _get_denoise_mode(self):
        """Convert UI denoise selection to libcamera enum value."""
        den = self.camera_denoise.currentText().strip().lower()
        if not den or den == "off":
            return None

        try:
            import libcamera
            nre = libcamera.controls.draft.NoiseReductionModeEnum
            return {
                "off": nre.Off,
                "fast": nre.Fast,
                "high_quality": nre.HighQuality,
                "highquality": nre.HighQuality,
            }.get(den)
        except Exception:
            # Fallback to integer values
            return {"off": 0, "fast": 1, "high_quality": 2, "highquality": 2}.get(den)

    # ---------------- Camera Controls ----------------

    def apply_camera_controls_live(self, skip_denoise=False):
        """Apply Picamera2/libcamera controls from the current UI values."""
        if not hasattr(self, "picam2") or self.picam2 is None:
            return

        try:
            supported = set(getattr(self.picam2, "camera_controls", {}).keys())
        except Exception:
            supported = set()

        def put(name: str, value):
            if supported and name not in supported:
                return
            controls[name] = value

        controls = {}

        # ExposureTime (µs)
        t = self.camera_shutter.text().strip()
        if t:
            put("ExposureTime", int(float(t)))

        # ExposureValue (EV compensation)
        ev = self.camera_ev.text().strip()
        if ev:
            put("ExposureValue", float(ev))

        # AnalogueGain / DigitalGain
        ag = self.camera_gain.text().strip()
        if ag:
            put("AnalogueGain", float(ag))

        dg = self.camera_digital_gain.text().strip() if hasattr(self, "camera_digital_gain") else ""
        if dg:
            put("DigitalGain", float(dg))

        # Contrast / Brightness / Saturation / Sharpness
        c = self.camera_contrast.text().strip()
        if c:
            put("Contrast", float(c))

        b = self.camera_brightness.text().strip()
        if b:
            put("Brightness", float(b))

        sat = self.camera_saturation.text().strip() if hasattr(self, "camera_saturation") else ""
        if sat:
            put("Saturation", float(sat))

        s = self.camera_sharpness.text().strip()
        if s:
            put("Sharpness", float(s))

        # White balance
        awb = self.camera_awbgains.text().strip()
        if awb:
            try:
                r_str, b_str = [x.strip() for x in awb.split(",")]
                r, bl = float(r_str), float(b_str)
                put("AwbEnable", False)
                put("ColourGains", (r, bl))
            except Exception:
                self.camera_status_label.setText("AWB Gains must be 'r,b' (e.g., 1.6,1.4)")
        else:
            put("AwbEnable", True)

        if not skip_denoise:
            den = self.camera_denoise.currentText().strip().lower()
            if den:
                nrm_val = None
                try:
                    import libcamera
                    nre = libcamera.controls.draft.NoiseReductionModeEnum
                    nrm_val = {
                        "off": nre.Off,
                        "fast": nre.Fast,
                        "high_quality": nre.HighQuality,
                        "highquality": nre.HighQuality,
                    }.get(den)
                except Exception:
                    nrm_val = {"off": 0, "fast": 1, "high_quality": 2, "highquality": 2}.get(den)

                if nrm_val is not None:
                    put("NoiseReductionMode", nrm_val)

        if not controls:
            return

        try:
            self.picam2.set_controls(controls)
        except Exception as e:
            self.camera_status_label.setText(f"Control apply failed: {e}")

    # ---------------- Live Preview ----------------

    def toggle_live_embedded(self):
        if self._live_running:
            self.stop_live_embedded()
        else:
            self.start_live_embedded()

    def start_live_embedded(self):
        if not hasattr(self, "picam2") or self.picam2 is None or not hasattr(self, "preview_thread"):
            self._init_embedded_camera()

        try:
            self.picam2.start()
            self.apply_camera_controls_live()
            self.preview_thread.start_stream()

            self.left_container.setVisible(True)
            self._live_running = True
            self.show_live_btn.setText("Stop Live")
            self.camera_status_label.setText("Live")
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._live_running = False
            self.show_live_btn.setText("Show Live")
            self.camera_status_label.setText(f"Live start failed: {e}")

    def stop_live_embedded(self):
        # Stop recording if active
        if self.is_recording:
            self.toggle_recording()

        try:
            if hasattr(self, "preview_thread") and self.preview_thread:
                self.preview_thread.stop_stream()
        finally:
            try:
                if hasattr(self, "picam2") and self.picam2:
                    self.picam2.stop()
            except Exception:
                pass

        self.left_container.setVisible(False)
        self._live_running = False
        self.show_live_btn.setText("Show Live")
        self.camera_status_label.setText("Idle")

    # ---------------- Recording & Analysis (NEW) ----------------

    def toggle_recording(self):
        """Start or stop the 10-second interval recording loop."""
        if not self._live_running:
            QMessageBox.warning(self, "Error", "Please start Live View before recording.")
            self.record_btn.setChecked(False)
            return

        self.is_recording = self.record_btn.isChecked()

        if self.is_recording:
            self.record_btn.setText("Stop Live Recording")
            self.recording_count = 0
            # Reset graph data for new session
            self.graph_data_counts = []
            self.graph_data_time = []
            self._update_graph()
            
            self.recording_timer.start()
        else:
            self.record_btn.setText("Start Live Recording (10s)")
            self.recording_timer.stop()

    def _recording_step(self):
        """Called every 10 seconds."""
        if not self._live_running:
            self.toggle_recording()
            return

        # 1. Resolve path with sequence/timestamp
        base_directory = self.selected_directory_label.text()
        if not base_directory or base_directory == "No directory selected":
            return # Skip if no folder

        current_date = datetime.now().strftime("%Y.%m.%d")
        current_time = datetime.now().strftime("%H.%M.%S")
        date_folder = os.path.join(base_directory, current_date)
        os.makedirs(date_folder, exist_ok=True)

        user_name = self.image_name.text() or "rec"
        # Append sequence or timestamp to filename to avoid overwrite
        filename = f"{current_date}_{current_time}_{user_name}_{self.recording_count:03d}.jpg"
        output_path = os.path.join(date_folder, filename)

        try:
            # 2. Capture (Reuse logic essentially, but keep array for model)
            self.apply_camera_controls_live(skip_denoise=True)
            self.picam2.options["quality"] = self.camera_quality.value()
            
            request = self.picam2.capture_request()
            
            # Save file
            request.save("main", output_path)
            
            # Get array for 'Model'
            image_array = request.make_array("main") 
            request.release()
            
            self.recording_count += 1
            self.camera_status_label.setText(f"Recorded: {filename}")

            # 3. Process Placeholder Model
            processed_img, detection_count = self._placeholder_model_inference(image_array)

            # 4. Update GUI (Model Preview)
            # processed_img is RGB numpy array. Convert to QImage.
            h, w, ch = processed_img.shape
            bytes_per_line = ch * w
            q_img = QImage(processed_img.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            
            pix = QPixmap.fromImage(q_img)
            self.model_preview.setPixmap(pix.scaled(
                self.model_preview.size(), 
                Qt.AspectRatioMode.KeepAspectRatio, 
                Qt.TransformationMode.FastTransformation
            ))

            # 5. Update Graph
            self.graph_data_time.append(self.recording_count)
            self.graph_data_counts.append(detection_count)
            self._update_graph()

        except Exception as e:
            print(f"Recording step failed: {e}")
            self.camera_status_label.setText(f"Rec Error: {e}")

    def _placeholder_model_inference(self, image_array):
        """
        Placeholder for AI Model.
        Input: RGB Numpy Array
        Output: (RGB Numpy Array with overlays, int detection_count)
        """
        # TODO: Replace this with real model inference later.
        
        # 1. Detection Count (Random)
        count = random.randint(0, 25)
        
        # 2. Image Processing (Just return original for now, maybe draw a box?)
        # Let's just return the original array. 
        # Ensure it's contiguous for QImage
        processed_array = np.ascontiguousarray(image_array)
        
        return processed_array, count

    def _update_graph(self):
        """Redraw the matplotlib graph."""
        self.ax.clear()
        self.ax.set_title("Detections over Time")
        self.ax.set_xlabel("Step")
        self.ax.set_ylabel("Count")
        self.ax.grid(True, linestyle='--', alpha=0.6)
        
        # Plot data
        self.ax.plot(self.graph_data_time, self.graph_data_counts, 'r-o', linewidth=2)
        
        # Refresh canvas
        self.canvas.draw()

    # ---------------- Capture (Manual) ----------------

    def select_save_directory(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Save Folder", self.selected_directory)
        if folder:
            self.selected_directory = folder
            self.selected_directory_label.setText(folder)

    def _resolve_output_path(self) -> str:
        base_directory = self.selected_directory_label.text()
        if not base_directory or base_directory == "No directory selected":
            QMessageBox.critical(self, "Error", "No directory selected")
            return None

        current_date = datetime.now().strftime("%Y.%m.%d")
        current_time = datetime.now().strftime("%H.%M.%S")
        date_folder = os.path.join(base_directory, current_date)
        os.makedirs(date_folder, exist_ok=True)

        image_name = self.image_name.text() or "image"
        output_path = os.path.join(date_folder, f"{current_date}_{current_time}_{image_name}.jpg")

        return output_path

    def _apply_channel_filter(self, frame):
        """Apply channel filter to numpy array (RGB input)."""
        channel = self.camera_channel.currentText().lower()
        if channel == "rgb":
            return frame
        elif channel == "red":
            filtered = np.zeros_like(frame)
            filtered[:, :, 0] = frame[:, :, 0]
            return filtered
        elif channel == "green":
            filtered = np.zeros_like(frame)
            filtered[:, :, 1] = frame[:, :, 1]
            return filtered
        elif channel == "blue":
            filtered = np.zeros_like(frame)
            filtered[:, :, 2] = frame[:, :, 2]
            return filtered
        elif channel == "grayscale":
            gray = np.mean(frame, axis=2).astype(np.uint8)
            return np.stack([gray, gray, gray], axis=2)
        return frame

    def capture_image(self):
        output = self._resolve_output_path()
        if not output:
            return

        self.apply_camera_controls_live(skip_denoise=True)
        self.picam2.options["quality"] = self.camera_quality.value()

        request = self.picam2.capture_request()

        # Check if channel filter needed
        channel = self.camera_channel.currentText().lower()
        if channel == "rgb":
            request.save("main", output)
        else:
            # Filtered capture - get array, filter, save manually
            frame = request.make_array("main")
            frame = frame[:, :, ::-1].copy()  # BGR -> RGB
            frame = self._apply_channel_filter(frame)

            from PIL import Image
            img = Image.fromarray(frame)
            img.save(output, quality=self.camera_quality.value())

        raw_path = None
        if self.camera_fullraw.isChecked():
            raw_path = output.replace('.jpg', '.dng')
            request.save_dng(raw_path)

        request.release()

        msg = f"Saved: {os.path.basename(output)}"
        if raw_path:
            msg += f" + DNG"
        self.camera_status_label.setText(msg)

    # ---------------- Settings ----------------

    def set_default_camera_settings(self):
        """Persist the current UI values as defaults in config.json."""
        self._set_cam("default_width", int(self.camera_width.value()))
        self._set_cam("default_height", int(self.camera_height.value()))
        self._set_cam("default_shutter", self._safe_text(self.camera_shutter))
        self._set_cam("default_contrast", self._safe_text(self.camera_contrast))
        self._set_cam("default_gain", self._safe_text(self.camera_gain))
        self._set_cam("default_awbgains", self._safe_text(self.camera_awbgains))
        self._set_cam("default_sharpness", self._safe_text(self.camera_sharpness))
        self._set_cam("default_quality", int(self.camera_quality.value()))
        self._set_cam("default_ev", self._safe_text(self.camera_ev))
        self._set_cam("default_prev_time", self._safe_text(self.camera_prev_time))
        self._set_cam("default_brightness", self._safe_text(self.camera_brightness))
        self._set_cam("default_denoise", self.camera_denoise.currentText().strip())

        try:
            if self.config_path:
                save_config_atomic(self.config, self.config_path)
            QMessageBox.information(self, "Saved", "Default camera settings saved to config.")
        except Exception as e:
            QMessageBox.warning(self, "Save Error", f"Could not save config:\n{e}")

    # ---------------- LED ----------------

    def toggle_LED(self):
        try:
            self.led_on = not self.led_on
            GPIO.output(self.led_pin, GPIO.HIGH if self.led_on else GPIO.LOW)
            self.LED_btn.setText("Turn LED Off" if self.led_on else "Turn LED On")
        except Exception as e:
            QMessageBox.critical(self, "GPIO Error", f"Failed to toggle LED:\n{e}")
            self.led_on = False
            try:
                GPIO.output(self.led_pin, GPIO.LOW)
            except Exception:
                pass
            self.LED_btn.setText("Turn LED On")

    # ---------------- Callbacks ----------------

    def _on_raw_preview_toggled(self, checked: bool):
        if hasattr(self, "preview_thread") and self.preview_thread:
            self.preview_thread.set_stream("raw" if checked else "lores")

    def _on_channel_changed(self, channel: str):
        if hasattr(self, "preview_thread") and self.preview_thread:
            self.preview_thread.set_channel(channel.lower())

    # ---------------- Cleanup ----------------

    def close(self):
        self.recording_timer.stop()
        try:
            if hasattr(self, "preview_thread") and self.preview_thread:
                self.preview_thread.stop_stream()
        except Exception:
            pass

        try:
            if hasattr(self, "picam2") and self.picam2:
                self.picam2.stop()
        except Exception:
            pass

        try:
            GPIO.output(self.led_pin, GPIO.LOW)
            GPIO.cleanup(self.led_pin)
        except Exception:
            pass

        super().close()