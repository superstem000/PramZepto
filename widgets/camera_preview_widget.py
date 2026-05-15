"""
Camera Preview Widget - Handles Picamera2 live preview and image capture.

This widget encapsulates all Picamera2 interaction:
- Camera initialization and configuration
- Live preview with threading
- Single image capture
- Capture to numpy array (for model analysis)

IMPORTANT: Only one Picamera2 instance can exist at a time. This widget
fully releases the camera when stop_preview() is called to allow other
widgets/screens to use the camera.
"""

import time
import numpy as np

from PyQt6.QtWidgets import QWidget, QLabel, QVBoxLayout, QSizePolicy
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QImage

from picamera2 import Picamera2
from libcamera import Transform

from camera_thread import CameraPreviewThread


class CameraPreviewWidget(QWidget):
    """Camera preview widget with Picamera2 integration."""

    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.config = config
        self.CAMERA = getattr(config, "camera", {})
        
        self.picam2 = None
        self.preview_thread = None
        self._is_running = False
        self._current_settings = {}
        
        self._build_ui()

    def _build_ui(self):
        """Build the preview UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.preview_label = QLabel()
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setStyleSheet("background-color: black;")
        self.preview_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, 
            QSizePolicy.Policy.Expanding
        )
        self.preview_label.setMinimumSize(1, 1)
        
        layout.addWidget(self.preview_label)

    def _release_camera(self):
        """
        Fully release all camera resources.
        This MUST be called before another widget/screen can use the camera.
        """
        print("CameraPreviewWidget: Releasing camera resources...")
        
        # 1. Stop preview thread first and wait for it
        if self.preview_thread is not None:
            try:
                self.preview_thread.stop_stream()
                # Wait for thread to finish (with timeout)
                if self.preview_thread.isRunning():
                    self.preview_thread.wait(3000)
                print("  - Preview thread stopped")
            except Exception as e:
                print(f"  - Error stopping preview thread: {e}")
            finally:
                # Disconnect signals to avoid issues
                try:
                    self.preview_thread.frame_ready.disconnect()
                    self.preview_thread.error.disconnect()
                except Exception:
                    pass
                self.preview_thread = None
        
        # 2. Stop the camera if running
        if self.picam2 is not None:
            if self._is_running:
                try:
                    self.picam2.stop()
                    print("  - Camera stopped")
                except Exception as e:
                    print(f"  - Error stopping camera: {e}")
            
            # 3. Close the camera to release hardware
            try:
                self.picam2.close()
                print("  - Camera closed")
            except Exception as e:
                print(f"  - Error closing camera: {e}")
            finally:
                self.picam2 = None
        
        self._is_running = False
        
        # 4. Small delay to ensure hardware is fully released
        time.sleep(0.3)
        print("CameraPreviewWidget: Camera resources released")

    def _init_camera(self, settings: dict):
        """Initialize Picamera2 with given settings."""
        # Always release first to ensure clean state
        self._release_camera()
        
        print("CameraPreviewWidget: Initializing camera...")
        
        try:
            # Create new Picamera2 instance
            self.picam2 = Picamera2()
            print("  - Picamera2 instance created")
            
            preview_w, preview_h = (1280, 720)
            capture_w = settings.get('width', 4056)
            capture_h = settings.get('height', 3040)

            transform = Transform(
                hflip=1 if settings.get('hflip', False) else 0,
                vflip=1 if settings.get('vflip', False) else 0
            )

            denoise_mode = self._get_denoise_mode(settings.get('denoise', 'off'))
            preview_controls = {}
            if denoise_mode is not None:
                preview_controls["NoiseReductionMode"] = denoise_mode

            # Dual stream: main=full res for capture, lores=low res for preview
            # Explicitly set main format to RGB888 for consistent array capture
            self.preview_config = self.picam2.create_preview_configuration(
                main={"size": (capture_w, capture_h), "format": "RGB888"},
                lores={"size": (preview_w, preview_h), "format": "RGB888"},
                raw={},
                transform=transform,
                controls=preview_controls
            )
            
            # Configure the camera
            self.picam2.configure(self.preview_config)
            print(f"  - Camera configured: main={capture_w}x{capture_h}, lores={preview_w}x{preview_h}")

            # Create preview thread
            stream = "raw" if settings.get('raw_preview', False) else "lores"
            self.preview_thread = CameraPreviewThread(
                self.picam2, fps=30, stream=stream, parent=self
            )
            self.preview_thread.frame_ready.connect(self._on_preview_frame)
            self.preview_thread.error.connect(self._on_camera_error)
            print("  - Preview thread created")

            self._current_settings = settings.copy()
            
            print("CameraPreviewWidget: Camera initialized successfully")
            return True
            
        except Exception as e:
            print(f"CameraPreviewWidget: Camera initialization FAILED: {e}")
            import traceback
            traceback.print_exc()
            # Clean up on failure
            self._release_camera()
            raise RuntimeError(f"Camera initialization failed: {e}")

    def _get_denoise_mode(self, denoise_str: str):
        """Convert denoise string to libcamera enum value."""
        if not denoise_str:
            return None
        den = str(denoise_str).strip().lower()
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
            return {"off": 0, "fast": 1, "high_quality": 2, "highquality": 2}.get(den)

    def _apply_controls(self, settings: dict, skip_denoise: bool = False):
        """Apply Picamera2 controls from settings."""
        if not self.picam2:
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
        t = str(settings.get('shutter', '')).strip()
        if t:
            try:
                put("ExposureTime", int(float(t)))
            except ValueError:
                pass

        # ExposureValue (EV compensation)
        ev = str(settings.get('ev', '')).strip()
        if ev:
            try:
                put("ExposureValue", float(ev))
            except ValueError:
                pass

        # AnalogueGain / DigitalGain
        ag = str(settings.get('gain', '')).strip()
        if ag:
            try:
                put("AnalogueGain", float(ag))
            except ValueError:
                pass

        dg = str(settings.get('digital_gain', '')).strip()
        if dg:
            try:
                put("DigitalGain", float(dg))
            except ValueError:
                pass

        # Contrast / Brightness / Saturation / Sharpness
        c = str(settings.get('contrast', '')).strip()
        if c:
            try:
                put("Contrast", float(c))
            except ValueError:
                pass

        b = str(settings.get('brightness', '')).strip()
        if b:
            try:
                put("Brightness", float(b))
            except ValueError:
                pass

        sat = str(settings.get('saturation', '')).strip()
        if sat:
            try:
                put("Saturation", float(sat))
            except ValueError:
                pass

        s = str(settings.get('sharpness', '')).strip()
        if s:
            try:
                put("Sharpness", float(s))
            except ValueError:
                pass

        # White balance
        awb = str(settings.get('awbgains', '')).strip()
        if awb:
            try:
                r_str, b_str = [x.strip() for x in awb.split(",")]
                r, bl = float(r_str), float(b_str)
                put("AwbEnable", False)
                put("ColourGains", (r, bl))
            except Exception:
                pass
        else:
            put("AwbEnable", True)

        if not skip_denoise:
            den = str(settings.get('denoise', '')).strip().lower()
            if den:
                nrm_val = self._get_denoise_mode(den)
                if nrm_val is not None:
                    put("NoiseReductionMode", nrm_val)

        if controls:
            try:
                self.picam2.set_controls(controls)
            except Exception as e:
                print(f"Control apply failed: {e}")

    def _on_preview_frame(self, qimg: QImage):
        """Handle incoming preview frame."""
        pix = QPixmap.fromImage(qimg)
        scaled = pix.scaled(
            self.preview_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.FastTransformation
        )
        self.preview_label.setPixmap(scaled)

    def _on_camera_error(self, msg: str):
        """Handle camera error."""
        print(f"Camera error: {msg}")

    # ========== Public Interface ==========

    def start_preview(self, settings: dict):
        """Start live preview with given settings."""
        print(f"CameraPreviewWidget: start_preview called, currently running: {self._is_running}")
        
        # Initialize camera (this handles cleanup internally)
        self._init_camera(settings)
        
        # Start camera and preview
        try:
            self.picam2.start()
            time.sleep(0.1)  # Brief pause for camera to stabilize
            self._apply_controls(settings)
            self.preview_thread.start_stream()
            self._is_running = True
            print("CameraPreviewWidget: Preview started successfully")
        except Exception as e:
            print(f"CameraPreviewWidget: Failed to start preview: {e}")
            import traceback
            traceback.print_exc()
            self._release_camera()
            raise RuntimeError(f"Failed to start camera preview: {e}")

    def stop_preview(self):
        """
        Stop live preview and FULLY RELEASE the camera.
        This allows other screens/widgets to use the camera.
        """
        print(f"CameraPreviewWidget: stop_preview called, currently running: {self._is_running}")
        
        # Fully release the camera so other screens can use it
        self._release_camera()
        
        # Clear the preview display
        self.preview_label.clear()
        self.preview_label.setStyleSheet("background-color: black;")
        
        print("CameraPreviewWidget: Preview stopped and camera released")

    def apply_settings(self, settings: dict):
        """
        Apply updated camera controls to the running camera.
        Called when the user clicks 'Apply Settings' during live preview.
        """
        if not self.picam2 or not self._is_running:
            raise RuntimeError("Camera not running — start live preview first")
        
        self._apply_controls(settings)
        self._current_settings.update(settings)
        
        # Update channel filter on preview thread if it changed
        if self.preview_thread:
            channel = str(settings.get('channel', 'RGB')).lower()
            self.preview_thread.set_channel(channel)
            
            # Update raw vs lores stream if toggled
            stream = "raw" if settings.get('raw_preview', False) else "lores"
            self.preview_thread.set_stream(stream)
        
        print("CameraPreviewWidget: Settings applied to live camera")

    def capture_image(self, output_path: str, settings: dict):
        """Capture and save image to file."""
        if not self.picam2 or not self._is_running:
            raise RuntimeError("Camera not running")

        self._apply_controls(settings, skip_denoise=True)
        self.picam2.options["quality"] = settings.get('quality', 95)

        request = self.picam2.capture_request()
        
        try:
            # Handle channel filter
            channel = str(settings.get('channel', 'RGB')).lower()
            if channel == "rgb":
                request.save("main", output_path)
            else:
                frame = request.make_array("main")
                # Ensure we have RGB (handle both RGB888 and XRGB formats)
                frame = self._ensure_rgb(frame)
                frame = self._apply_channel_filter(frame, channel)

                from PIL import Image
                img = Image.fromarray(frame)
                img.save(output_path, quality=settings.get('quality', 95))

            # Save raw if requested
            if settings.get('save_raw', False):
                raw_path = output_path.replace('.jpg', '.dng')
                request.save_dng(raw_path)
        finally:
            request.release()

    def capture_to_array(self, settings: dict) -> np.ndarray:
        """Capture image and return as numpy array (RGB)."""
        if not self.picam2:
            print("capture_to_array: picam2 is None")
            return None
            
        if not self._is_running:
            print("capture_to_array: camera is not running")
            return None

        try:
            self._apply_controls(settings, skip_denoise=True)

            request = self.picam2.capture_request()
            try:
                frame = request.make_array("main")
            finally:
                request.release()

            # Ensure we have 3-channel RGB
            frame = self._ensure_rgb(frame)

            # Apply channel filter if needed
            channel = str(settings.get('channel', 'RGB')).lower()
            if channel != 'rgb':
                frame = self._apply_channel_filter(frame, channel)

            return np.ascontiguousarray(frame)
            
        except Exception as e:
            print(f"capture_to_array failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def flush_and_capture(self, settings: dict, discard: int = 3) -> np.ndarray:
        """
        Discard stale buffered frames, then capture a fresh one.
        
        This is critical for autofocus: after a Z move the buffer
        still holds frames from the previous position. We discard
        a few to ensure the returned image matches the current Z.
        
        Args:
            settings: Camera settings dict.
            discard:  Number of frames to throw away before the real capture.
            
        Returns:
            RGB numpy array from the main (full-res) stream, or None.
        """
        if not self.picam2 or not self._is_running:
            return None

        try:
            # Discard stale frames
            for i in range(discard):
                req = self.picam2.capture_request()
                req.release()

            # Now capture the real frame
            self._apply_controls(settings, skip_denoise=True)
            request = self.picam2.capture_request()
            try:
                frame = request.make_array("main")
            finally:
                request.release()

            frame = self._ensure_rgb(frame)

            channel = str(settings.get('channel', 'RGB')).lower()
            if channel != 'rgb':
                frame = self._apply_channel_filter(frame, channel)

            return np.ascontiguousarray(frame)

        except Exception as e:
            print(f"flush_and_capture failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _ensure_rgb(self, frame: np.ndarray) -> np.ndarray:
        """Ensure frame is 3-channel RGB format."""
        if frame is None:
            return None
            
        # Handle different formats
        if len(frame.shape) == 2:
            # Grayscale - convert to RGB
            return np.stack([frame, frame, frame], axis=2)
        elif len(frame.shape) == 3:
            if frame.shape[2] == 4:
                # XRGB or XBGR - drop alpha channel
                # Picamera2 usually returns XBGR, so we take channels 2,1,0
                return frame[:, :, 2::-1].copy()  # XBGR -> RGB
            elif frame.shape[2] == 3:
                # Already 3 channels - should be RGB since we configured RGB888
                return frame[:, :, ::-1].copy()
        
        return frame

    def _apply_channel_filter(self, frame: np.ndarray, channel: str) -> np.ndarray:
        """Apply channel filter to numpy array (RGB input)."""
        if channel == "red":
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

    def is_running(self) -> bool:
        """Check if preview is currently running."""
        return self._is_running

    def close(self):
        """Clean up resources."""
        print("CameraPreviewWidget: close() called")
        self._release_camera()
        super().close()