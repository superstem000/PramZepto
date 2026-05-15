from PyQt6.QtWidgets import (
    QApplication, QLabel, QPushButton, QFileDialog, QVBoxLayout,
    QWidget, QHBoxLayout, QFrame, QSplashScreen, QGridLayout, QTextEdit,
    QGroupBox, QFormLayout, QComboBox, QDoubleSpinBox, QSpinBox, QSlider, QLineEdit, QSizePolicy, QCheckBox, QComboBox,
    QMessageBox, QMainWindow, QScrollArea
)
from PyQt6.QtGui import QPixmap, QCursor, QGuiApplication
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal, QSize, QCoreApplication, QObject
from config import save_config_atomic
from datetime import datetime
from gpiozero import DigitalInputDevice
import subprocess, time, tempfile, signal, os

from stepper_motors_widget import StepperMotorsWidget
from stepper_motors_hardware import StepperMotorHardware
from optics_widget import OpticsWidget
from move_planner_widget import MovePlannerWidget
from camera_widget import CameraWidget

class NanoCounter(QWidget):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.config_path = config.cfg_path

        self.THEME  = getattr(self.config, "theme", {})  
        self.MOTOR  = getattr(self.config, "motor", {})
        self.CAMERA = getattr(self.config, "camera", {})

        self.mc = StepperMotorHardware(self.config)
        self._initUI()

    def _initUI(self):
        self.setWindowTitle("NanoCounter")
        self.zoomed_index = None
        self.zoomed_view = None

        # Move the window to the right side of the monitor
        screen_geometry = QGuiApplication.primaryScreen().geometry()
        screen_width, screen_height = screen_geometry.width(), screen_geometry.height()
        self.window_width = screen_width
        self.window_height = screen_height - 70
        # self.resize(self.window_width, self.window_height) 
        # self.move(0 - 20, 30)
        self.setWindowState(Qt.WindowState.WindowMaximized)
        self.setStyleSheet(f"""
            #scrollContent {{
                border: none;
                background-color: {self.THEME['bg_color']};
            }}
            QWidget {{
                background-color: {self.THEME['bg_color']};
                font-family: {self.THEME['font_family']};
                font-size: {self.THEME['font_size']};
                color: {self.THEME['font_color']};
            }}
            QPushButton {{
                background-color: {self.THEME['button_bg']};
                color: {self.THEME['font_color']};
                border: {self.THEME['button_border']};
                border-radius: {self.THEME['button_radius']};
                padding: 5px 12px;
            }}
            QPushButton:hover {{
                background-color: {self.THEME['button_hover']};
            }}
            QPushButton:pressed {{
                background-color: {self.THEME['button_pressed']};
            }}
            QPushButton:disabled {{
                background-color: #d3d3d3;  /* Light gray */
                color: #a0a0a0;            /* Dimmed text color */
                border: 1px solid #999;    /* Optional border */
            }}
            QGroupBox {{
                background-color: {self.THEME['panel_bg']};
                border: {self.THEME['border_width']} solid {self.THEME['border_color']};
                border-radius: 5px;
                margin-top: 10px;
                padding: 10px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                background-color: {self.THEME['groupbox_title_bg']};
                font-weight: bold;
            }}
        """)

        # LEFT PANEL
        self.left_panel = QWidget()
        self.left_panel.setObjectName("leftPanel")
        self.left_panel_layout = QVBoxLayout(self.left_panel)
        self.left_content = QWidget()
        self.left_content_layout = QGridLayout(self.left_content)
        self.left_content_layout.setContentsMargins(0, 0, 0, 0)
        self.left_content_layout.setRowStretch(0, 1)
        self.left_content_layout.setColumnStretch(0, 1)
        self.left_panel_layout.addWidget(self.left_content)
        self.left_content.setVisible(False) 
        self.left_panel.setStyleSheet(f"""
        Qwidget#leftPanel {{
            background-color: {self.THEME['panel_bg']};
            }}
        """)

        # RIGHT PANEL (SCROLLABLE)
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setContentsMargins(0,0,0,0)
        self.scroll_area.setObjectName("rightScroll")
        self.scroll_area.setStyleSheet(f"""
        QScrollArea#rightScroll {{
            border: none;
            border-left: 3px solid gray;
            background-color: {self.THEME['bg_color']}
        }}
        """)
        
        self.scroll_content = QWidget()
        self.function_layout = QVBoxLayout(self.scroll_content)
        self.function_layout.setContentsMargins(0,0,0,0)
        self.function_layout.setSpacing(6)
        
        self.scroll_area.setWidget(self.scroll_content)
        self.scroll_area.setFixedWidth(600)

        #MAIN LAYOUT
        self.main_layout = QHBoxLayout()
        self.main_layout.addWidget(self.left_panel, stretch=4)
        self.main_layout.addWidget(self.scroll_area, stretch=1)
        self.main_layout.setContentsMargins(0,0,0,0)
        self.setLayout(self.main_layout)

        self._stepper_motors = StepperMotorsWidget(self.function_layout, self.config, self.mc)
        self._move_planner = MovePlannerWidget(self.function_layout, self.config, self._stepper_motors)
        self._camera = CameraWidget(self.left_content, self.left_content_layout, self.function_layout, self.config)
        # self._optics = OpticsWidget(self.function_layout, self.config)

        self.function_layout.addStretch()

    def closeEvent(self, event):
        try:
            self.mc.close()
        except Exception:
            pass
        super().closeEvent(event)

