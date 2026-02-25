"""
Biomechanics GUI - Interactive Video Player with Skeletal Overlay
=================================================================
PyQt5 application for analyzing biomechanical motion data.
- Video playback with marker overlay and skeleton connections
- Click any marker to select it
- Real-time kinematics display (position, velocity, acceleration)
- Embedded matplotlib graphs
"""

import sys
import os
import numpy as np
import cv2
from pathlib import Path

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSlider, QGroupBox, QGridLayout, QComboBox,
    QSplitter, QFrame, QSizePolicy, QToolTip
)
from PyQt5.QtCore import Qt, QTimer, QPoint
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor, QPainter, QPen, QCursor

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# ──────────────────────────────────────────────
# Configuration  –  EDIT THESE TO SWAP VIDEO / TRACKING DATA
# ──────────────────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parent
DATA_DIR  = BASE_DIR / "data"

VIDEO_FILE  = DATA_DIR / "vid.mp4"                           # source video
TRC_PX_FILE = DATA_DIR / "vid_Sports2D_px_person00.trc"      # pixel-space marker positions
TRC_M_FILE  = DATA_DIR / "vid_Sports2D_m_person00.trc"       # meter-space marker positions
MOT_FILE    = DATA_DIR / "vid_Sports2D_angles_person00.mot"  # joint angles (.mot)

# Skeleton connections (pairs of marker names)
SKELETON_CONNECTIONS = [
    ("Hip", "RHip"), ("Hip", "LHip"),
    ("RHip", "RKnee"), ("RKnee", "RAnkle"),
    ("RAnkle", "RBigToe"), ("RAnkle", "RHeel"),
    ("LHip", "LKnee"), ("LKnee", "LAnkle"),
    ("LAnkle", "LBigToe"), ("LAnkle", "LHeel"),
    ("Hip", "Neck"), ("Neck", "Head"), ("Head", "Nose"),
    ("Neck", "RShoulder"), ("Neck", "LShoulder"),
    ("RShoulder", "RElbow"), ("RElbow", "RWrist"),
    ("LShoulder", "LElbow"), ("LElbow", "LWrist"),
]

# Marker to joint mapping
MARKER_JOINT_MAP = {
    "RAnkle": "right ankle", "RKnee": "right knee", "RHip": "right hip",
    "RElbow": "right elbow", "RWrist": "right wrist",
    "LAnkle": "left ankle", "LKnee": "left knee", "LHip": "left hip",
    "LElbow": "left elbow", "LWrist": "left wrist",
    "RShoulder": "right shoulder", "LShoulder": "left shoulder",
    "Hip": "right hip", "Neck": "neck",
    "Head": "head", "Nose": "nose",
    "RBigToe": "right ankle", "RSmallToe": "right ankle", "RHeel": "right ankle",
    "LBigToe": "left ankle", "LSmallToe": "left ankle", "LHeel": "left ankle",
}

# Joint angle display: joint_name -> (parent_marker, vertex_marker, child_marker)
JOINT_ANGLE_MARKERS = {
    "right knee":     ("RHip",      "RKnee",    "RAnkle"),
    "left knee":      ("LHip",      "LKnee",    "LAnkle"),
    "right hip":      ("Neck",      "RHip",     "RKnee"),
    "left hip":       ("Neck",      "LHip",     "LKnee"),
    "right shoulder": ("Neck",      "RShoulder", "RElbow"),
    "left shoulder":  ("Neck",      "LShoulder", "LElbow"),
    "right elbow":    ("RShoulder", "RElbow",   "RWrist"),
    "left elbow":     ("LShoulder", "LElbow",   "LWrist"),
    "right ankle":    ("RKnee",     "RAnkle",   "RBigToe"),
    "left ankle":     ("LKnee",     "LAnkle",   "LBigToe"),
}

MARKER_COLORS = {
    "Hip": "#e63946", "RHip": "#e63946", "LHip": "#e63946",
    "RKnee": "#457b9d", "RAnkle": "#457b9d", "RBigToe": "#457b9d",
    "RSmallToe": "#457b9d", "RHeel": "#457b9d",
    "LKnee": "#2a9d8f", "LAnkle": "#2a9d8f", "LBigToe": "#2a9d8f",
    "LSmallToe": "#2a9d8f", "LHeel": "#2a9d8f",
    "Neck": "#f4a261", "Head": "#f4a261", "Nose": "#f4a261",
    "RShoulder": "#e9c46a", "RElbow": "#e9c46a", "RWrist": "#e9c46a",
    "LShoulder": "#264653", "LElbow": "#264653", "LWrist": "#264653",
}

HIT_RADIUS = 15  # pixels for click detection
DATA_FPS = 250   # original tracking data frame rate (real-time speed)


# ──────────────────────────────────────────────
# Data Loading
# ──────────────────────────────────────────────
def parse_trc(filepath):
    """Parse a .trc file into arrays. Returns (marker_names, time, positions_dict)."""
    with open(filepath, "r") as f:
        lines = f.readlines()

    meta = lines[2].strip().split("\t")
    marker_names_raw = lines[3].strip().split("\t")
    marker_names = [n for n in marker_names_raw[2:] if n.strip()]

    cols = ["Frame", "Time"]
    for m in marker_names:
        cols.extend([f"{m}_X", f"{m}_Y", f"{m}_Z"])

    data_lines = lines[5:]
    rows = []
    for line in data_lines:
        parts = line.strip().split("\t")
        if len(parts) < 3:
            continue
        row = [float(v) for v in parts[:len(cols)]]
        if len(row) == len(cols):
            rows.append(row)

    data = np.array(rows)
    time_arr = data[:, 1]

    positions = {}
    for i, m in enumerate(marker_names):
        col_start = 2 + i * 3
        positions[m] = {
            "x": data[:, col_start],
            "y": data[:, col_start + 1],
            "z": data[:, col_start + 2],
        }

    return marker_names, time_arr, positions


def parse_mot(filepath):
    """Parse a .mot file. Returns (joint_names, time, angles_dict)."""
    with open(filepath, "r") as f:
        lines = f.readlines()

    header_end = 0
    for i, line in enumerate(lines):
        if line.strip().lower() == "endheader":
            header_end = i
            break

    col_line = lines[header_end + 1].strip().split("\t")
    data_lines = lines[header_end + 2:]

    rows = []
    for line in data_lines:
        parts = line.strip().split("\t")
        if len(parts) < 2:
            continue
        row = [float(v) for v in parts]
        if len(row) == len(col_line):
            rows.append(row)

    data = np.array(rows)
    time_arr = data[:, 0]
    joint_names = col_line[1:]

    angles = {}
    for i, j in enumerate(joint_names):
        angles[j] = data[:, i + 1]

    return joint_names, time_arr, angles


def compute_kinematics(positions_m, time_arr):
    """Pre-compute velocity and acceleration for all markers in meter space."""
    dt = np.mean(np.diff(time_arr))
    kinematics = {}
    for marker, pos in positions_m.items():
        x, y = pos["x"], pos["y"]
        vx = np.gradient(x, dt)
        vy = np.gradient(y, dt)
        speed = np.sqrt(vx**2 + vy**2)
        ax = np.gradient(vx, dt)
        ay = np.gradient(vy, dt)
        accel = np.sqrt(ax**2 + ay**2)
        kinematics[marker] = {
            "x": x, "y": y,
            "vx": vx, "vy": vy, "speed": speed,
            "ax": ax, "ay": ay, "accel": accel,
        }
    return kinematics, dt


def compute_angular_kinematics(angles, time_arr):
    """Pre-compute angular velocity and acceleration for all joints.
    Stores display angle as interior angle (0-180) using 180-theta correction,
    but computes omega/alpha from raw theta for correct derivatives."""
    dt = np.mean(np.diff(time_arr))
    ang_kin = {}
    for joint, theta in angles.items():
        # Derivatives from raw angles (sign = rotation direction)
        omega = np.gradient(theta, dt)
        alpha = np.gradient(omega, dt)
        # Display angle: 180-theta, then normalize to interior angle [0, 180]
        display = 180.0 - theta
        display = display % 360
        display = np.where(display > 180, 360 - display, display)
        ang_kin[joint] = {
            "angle": display,
            "omega": omega,
            "alpha": alpha,
        }
    return ang_kin, dt


# ──────────────────────────────────────────────
# Video Display Widget
# ──────────────────────────────────────────────
class VideoWidget(QLabel):
    """QLabel that displays video frames with marker overlay and handles clicks."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(480, 384)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)

        self.marker_positions = {}  # {name: (x, y)} for current frame
        self.marker_names = []
        self.selected_marker = None
        self.on_marker_selected = None  # callback
        self._scale = 1.0
        self._offset_x = 0
        self._offset_y = 0
        self._video_w = 600
        self._video_h = 480

    def set_frame(self, frame, marker_positions, marker_names, angle_info=None):
        """Draw a video frame with markers and skeleton overlay.
        angle_info: optional dict with keys 'marker_name', 'joint_pos', 'parent_pos', 'child_pos', 'display_val'
        """
        self.marker_positions = marker_positions
        self.marker_names = marker_names

        h, w = frame.shape[:2]
        self._video_w = w
        self._video_h = h

        # Draw skeleton connections
        for m1, m2 in SKELETON_CONNECTIONS:
            if m1 in marker_positions and m2 in marker_positions:
                p1 = marker_positions[m1]
                p2 = marker_positions[m2]
                if p1[0] != 0 and p1[1] != 0 and p2[0] != 0 and p2[1] != 0:
                    cv2.line(frame, (int(p1[0]), int(p1[1])),
                             (int(p2[0]), int(p2[1])), (200, 200, 200), 2)

        # Draw markers
        for name in marker_names:
            if name not in marker_positions:
                continue
            px, py = marker_positions[name]
            if px == 0 and py == 0:
                continue

            color_hex = MARKER_COLORS.get(name, "#ffffff")
            r = int(color_hex[1:3], 16)
            g = int(color_hex[3:5], 16)
            b = int(color_hex[5:7], 16)

            radius = 7
            thickness = -1
            if name == self.selected_marker:
                radius = 11
                cv2.circle(frame, (int(px), int(py)), radius + 3, (255, 255, 255), 2)

            cv2.circle(frame, (int(px), int(py)), radius, (b, g, r), thickness)

        # Draw selected marker name label (on top of markers/skeleton)
        if self.selected_marker and self.selected_marker in marker_positions:
            sx, sy = marker_positions[self.selected_marker]
            if sx != 0 and sy != 0:
                color_hex = MARKER_COLORS.get(self.selected_marker, "#ffffff")
                cr = int(color_hex[1:3], 16)
                cg = int(color_hex[3:5], 16)
                cb = int(color_hex[5:7], 16)
                cv2.putText(frame, self.selected_marker, (int(sx) + 14, int(sy) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
                cv2.putText(frame, self.selected_marker, (int(sx) + 14, int(sy) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (cb, cg, cr), 1)

        # Draw angle arc + text (topmost z-order)
        if angle_info:
            vx, vy = angle_info['joint_pos']
            px, py = angle_info['parent_pos']
            cx, cy = angle_info['child_pos']
            display_val = angle_info['display_val']
            ang1 = np.degrees(np.arctan2(-(py - vy), px - vx))
            ang2 = np.degrees(np.arctan2(-(cy - vy), cx - vx))
            arc_radius = 20
            start_angle = min(ang1, ang2)
            end_angle = max(ang1, ang2)
            if end_angle - start_angle > 180:
                start_angle, end_angle = end_angle, start_angle + 360
            cv2.ellipse(frame, (int(vx), int(vy)), (arc_radius, arc_radius),
                        0, -end_angle, -start_angle, (0, 255, 255), 2)
            mid_ang = np.radians((ang1 + ang2) / 2)
            tx = int(vx + (arc_radius + 14) * np.cos(mid_ang))
            ty = int(vy - (arc_radius + 14) * np.sin(mid_ang))
            cv2.putText(frame, f"{display_val:.0f}deg", (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 3)
            cv2.putText(frame, f"{display_val:.0f}deg", (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        # Convert to QPixmap and display
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

        # Scale to fit widget
        scaled = pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._scale = scaled.width() / w
        self._offset_x = (self.width() - scaled.width()) // 2
        self._offset_y = (self.height() - scaled.height()) // 2
        self.setPixmap(scaled)

    def _widget_to_video(self, wx, wy):
        """Convert widget coordinates to video pixel coordinates."""
        vx = (wx - self._offset_x) / self._scale
        vy = (wy - self._offset_y) / self._scale
        return vx, vy

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            vx, vy = self._widget_to_video(event.x(), event.y())
            best = None
            best_dist = HIT_RADIUS
            for name, (px, py) in self.marker_positions.items():
                if px == 0 and py == 0:
                    continue
                dist = np.sqrt((vx - px)**2 + (vy - py)**2)
                if dist < best_dist:
                    best_dist = dist
                    best = name
            if best:
                self.selected_marker = best
                if self.on_marker_selected:
                    self.on_marker_selected(best)

    def mouseMoveEvent(self, event):
        vx, vy = self._widget_to_video(event.x(), event.y())
        for name, (px, py) in self.marker_positions.items():
            if px == 0 and py == 0:
                continue
            dist = np.sqrt((vx - px)**2 + (vy - py)**2)
            if dist < HIT_RADIUS:
                QToolTip.showText(event.globalPos(), name, self)
                return
        QToolTip.hideText()


# ──────────────────────────────────────────────
# Graph Widget
# ──────────────────────────────────────────────
class GraphWidget(FigureCanvas):
    """Embedded matplotlib canvas showing kinematics graphs."""

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(5, 6), dpi=100)
        self.fig.set_facecolor("#1e1e2e")
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.axes = []
        self._vlines = []

    def plot_kinematics(self, marker_name, time_m, kin, joint_name, ang_kin, time_mot, current_frame):
        """Plot X/Y velocity, X/Y acceleration, angular velocity, angular acceleration."""
        self.fig.clear()
        self.axes = []
        self._vlines = []

        current_time = time_m[current_frame] if current_frame < len(time_m) else time_m[-1]

        has_angular = joint_name and joint_name in ang_kin
        n_plots = 4 if has_angular else 2

        # Plot 1: X & Y Velocity
        ax1 = self.fig.add_subplot(n_plots, 1, 1)
        ax1.plot(time_m, kin["vx"], color="#89b4fa", alpha=0.9, linewidth=1.2, label="Vx")
        ax1.plot(time_m, kin["vy"], color="#a6e3a1", alpha=0.9, linewidth=1.2, label="Vy")
        vl = ax1.axvline(current_time, color="white", linestyle="--", alpha=0.7, linewidth=1)
        self._vlines.append(vl)
        ax1.set_ylabel("Velocity (m/s)", fontsize=8)
        ax1.set_title(f"{marker_name} - Linear Velocity", fontsize=9, color="white", pad=4)
        ax1.legend(fontsize=7, loc="upper right", framealpha=0.3)
        self._style_axis(ax1)

        # Plot 2: X & Y Acceleration
        ax2 = self.fig.add_subplot(n_plots, 1, 2)
        ax2.plot(time_m, kin["ax"], color="#f38ba8", alpha=0.9, linewidth=1.2, label="Ax")
        ax2.plot(time_m, kin["ay"], color="#fab387", alpha=0.9, linewidth=1.2, label="Ay")
        vl = ax2.axvline(current_time, color="white", linestyle="--", alpha=0.7, linewidth=1)
        self._vlines.append(vl)
        ax2.set_ylabel("Accel (m/s^2)", fontsize=8)
        ax2.set_title(f"{marker_name} - Linear Acceleration", fontsize=9, color="white", pad=4)
        ax2.legend(fontsize=7, loc="upper right", framealpha=0.3)
        self._style_axis(ax2)

        all_axes = [ax1, ax2]

        if has_angular:
            n = min(len(time_mot), len(ang_kin[joint_name]["omega"]))

            # Plot 3: Angular Velocity
            ax3 = self.fig.add_subplot(n_plots, 1, 3)
            ax3.plot(time_mot[:n], ang_kin[joint_name]["omega"][:n],
                     color="#cba6f7", alpha=0.9, linewidth=1.2)
            vl = ax3.axvline(current_time, color="white", linestyle="--", alpha=0.7, linewidth=1)
            self._vlines.append(vl)
            ax3.set_ylabel("Ang Vel (deg/s)", fontsize=8)
            ax3.set_title(f"{joint_name.title()} - Angular Velocity", fontsize=9, color="white", pad=4)
            self._style_axis(ax3)
            all_axes.append(ax3)

            # Plot 4: Angular Acceleration
            ax4 = self.fig.add_subplot(n_plots, 1, 4)
            ax4.plot(time_mot[:n], ang_kin[joint_name]["alpha"][:n],
                     color="#f9e2af", alpha=0.9, linewidth=1.2)
            vl = ax4.axvline(current_time, color="white", linestyle="--", alpha=0.7, linewidth=1)
            self._vlines.append(vl)
            ax4.set_ylabel("Ang Acc (deg/s^2)", fontsize=8)
            ax4.set_title(f"{joint_name.title()} - Angular Acceleration", fontsize=9, color="white", pad=4)
            self._style_axis(ax4)
            all_axes.append(ax4)

        all_axes[-1].set_xlabel("Time (s)", fontsize=8)
        self.axes = all_axes
        self.fig.tight_layout(pad=1.0, h_pad=0.8)
        self.draw()

    def update_vline(self, time_val):
        """Update the vertical line position without full redraw."""
        for vl in self._vlines:
            vl.set_xdata([time_val, time_val])
        self.draw_idle()

    def _style_axis(self, ax):
        ax.set_facecolor("#2a2a3e")
        ax.tick_params(colors="white", labelsize=7)
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_color("#555")
        ax.grid(True, alpha=0.2, color="white")


# ──────────────────────────────────────────────
# Main Window
# ──────────────────────────────────────────────
class BiomechanicsGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Biomechanics Motion Analysis")
        self.setMinimumSize(1200, 700)
        self.setStyleSheet(self._get_stylesheet())

        # Load data
        self._load_data()

        # Video capture
        self.cap = cv2.VideoCapture(str(VIDEO_FILE))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.current_frame_idx = 0
        self.is_playing = False

        # Timer for playback
        self.timer = QTimer()
        self.timer.timeout.connect(self._next_frame)

        # Selected marker
        self.selected_marker = None
        self.graph_marker = None  # marker currently graphed
        self.show_trajectory = False
        self.show_relative_trajectory = False

        # Build UI
        self._build_ui()

        # Show first frame
        self._show_frame(0)

    def _load_data(self):
        """Load all data files."""
        self.marker_names, self.time_px, self.pos_px = parse_trc(TRC_PX_FILE)
        _, self.time_m, self.pos_m = parse_trc(TRC_M_FILE)
        self.joint_names, self.time_mot, self.angles = parse_mot(MOT_FILE)

        self.kin, self.dt_m = compute_kinematics(self.pos_m, self.time_m)
        self.ang_kin, self.dt_mot = compute_angular_kinematics(self.angles, self.time_mot)

        self.n_frames_px = len(self.time_px)
        self.n_frames_m = len(self.time_m)
        self.n_frames_mot = len(self.time_mot)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 8)

        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # ── Left panel: Video ──
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)

        self.video_widget = VideoWidget()
        self.video_widget.on_marker_selected = self._on_marker_selected
        left_layout.addWidget(self.video_widget, 1)

        # Controls
        controls = QWidget()
        ctrl_layout = QHBoxLayout(controls)
        ctrl_layout.setContentsMargins(4, 4, 4, 4)

        self.btn_play = QPushButton("Play")
        self.btn_play.setFixedWidth(70)
        self.btn_play.clicked.connect(self._toggle_play)
        ctrl_layout.addWidget(self.btn_play)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, max(self.total_frames - 1, 0))
        self.slider.valueChanged.connect(self._slider_changed)
        ctrl_layout.addWidget(self.slider, 1)

        self.lbl_frame = QLabel("0 / 0")
        self.lbl_frame.setFixedWidth(120)
        self.lbl_frame.setAlignment(Qt.AlignCenter)
        ctrl_layout.addWidget(self.lbl_frame)

        ctrl_layout.addWidget(QLabel("Speed:"))
        self.combo_speed = QComboBox()
        self.combo_speed.addItems(["0.1x", "0.25x", "0.5x", "1x", "2x"])
        self.combo_speed.setCurrentText("1x")
        self.combo_speed.currentTextChanged.connect(self._speed_changed)
        self.combo_speed.setFixedWidth(70)
        ctrl_layout.addWidget(self.combo_speed)

        left_layout.addWidget(controls)
        splitter.addWidget(left_widget)

        # ── Right panel: Data + Graph ──
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(4, 0, 0, 0)

        # Title
        title = QLabel("Kinematics Data")
        title.setFont(QFont("Segoe UI", 13, QFont.Bold))
        title.setStyleSheet("color: #cdd6f4; padding: 4px;")
        right_layout.addWidget(title)

        # Selected marker label
        self.lbl_selected = QLabel("Click a marker on the video to select it")
        self.lbl_selected.setStyleSheet("color: #a6adc8; font-size: 11px; padding: 2px;")
        right_layout.addWidget(self.lbl_selected)

        # Kinematics data grid
        self.data_group = QGroupBox("Current Frame Data")
        self.data_group.setStyleSheet("QGroupBox { color: #cdd6f4; font-weight: bold; }")
        data_grid = QGridLayout(self.data_group)
        data_grid.setSpacing(4)

        labels = [
            "Position (m):", "Linear Velocity (m/s):", "Linear Acceleration (m/s^2):",
            "Joint Angle (deg):", "Angular Velocity (deg/s):", "Angular Accel (deg/s^2):"
        ]
        self.data_values = []
        for i, lbl in enumerate(labels):
            l = QLabel(lbl)
            l.setStyleSheet("color: #bac2de; font-size: 10px;")
            v = QLabel("--")
            v.setStyleSheet("color: #cdd6f4; font-size: 11px; font-weight: bold;")
            v.setTextInteractionFlags(Qt.TextSelectableByMouse)
            data_grid.addWidget(l, i, 0)
            data_grid.addWidget(v, i, 1)
            self.data_values.append(v)

        right_layout.addWidget(self.data_group)

        # Trajectory toggle button
        self.btn_trajectory = QPushButton("Show Trajectory")
        self.btn_trajectory.setCheckable(True)
        self.btn_trajectory.setEnabled(False)
        self.btn_trajectory.toggled.connect(self._toggle_trajectory)
        self.btn_trajectory.setStyleSheet(
            "QPushButton { background-color: #45475a; color: #cdd6f4; font-weight: bold; "
            "padding: 8px; border-radius: 6px; border: 1px solid #585b70; } "
            "QPushButton:hover { background-color: #585b70; } "
            "QPushButton:checked { background-color: #89b4fa; color: #1e1e2e; } "
            "QPushButton:disabled { background-color: #313244; color: #6c7086; }"
        )
        right_layout.addWidget(self.btn_trajectory)

        # Relative trajectory toggle button
        self.btn_rel_trajectory = QPushButton("Show Relative Trajectory")
        self.btn_rel_trajectory.setCheckable(True)
        self.btn_rel_trajectory.setEnabled(False)
        self.btn_rel_trajectory.toggled.connect(self._toggle_relative_trajectory)
        self.btn_rel_trajectory.setStyleSheet(
            "QPushButton { background-color: #45475a; color: #cdd6f4; font-weight: bold; "
            "padding: 8px; border-radius: 6px; border: 1px solid #585b70; } "
            "QPushButton:hover { background-color: #585b70; } "
            "QPushButton:checked { background-color: #a6e3a1; color: #1e1e2e; } "
            "QPushButton:disabled { background-color: #313244; color: #6c7086; }"
        )
        right_layout.addWidget(self.btn_rel_trajectory)

        # Graph canvas
        self.graph_widget = GraphWidget()
        self.graph_widget.setMinimumHeight(300)
        right_layout.addWidget(self.graph_widget, 1)

        splitter.addWidget(right_widget)
        splitter.setSizes([700, 500])

    def _get_stylesheet(self):
        return """
        QMainWindow { background-color: #1e1e2e; }
        QWidget { background-color: #1e1e2e; color: #cdd6f4; }
        QLabel { color: #cdd6f4; }
        QPushButton {
            background-color: #45475a; color: #cdd6f4; border: 1px solid #585b70;
            padding: 6px 12px; border-radius: 4px; font-size: 11px;
        }
        QPushButton:hover { background-color: #585b70; }
        QPushButton:pressed { background-color: #6c7086; }
        QSlider::groove:horizontal {
            background: #313244; height: 6px; border-radius: 3px;
        }
        QSlider::handle:horizontal {
            background: #89b4fa; width: 14px; margin: -4px 0;
            border-radius: 7px;
        }
        QSlider::sub-page:horizontal { background: #89b4fa; border-radius: 3px; }
        QComboBox {
            background-color: #45475a; color: #cdd6f4; border: 1px solid #585b70;
            padding: 4px; border-radius: 4px;
        }
        QComboBox::drop-down { border: none; }
        QComboBox QAbstractItemView {
            background-color: #313244; color: #cdd6f4; selection-background-color: #585b70;
        }
        QGroupBox {
            border: 1px solid #45475a; border-radius: 6px; margin-top: 10px; padding: 10px;
        }
        QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
        QSplitter::handle { background-color: #45475a; width: 3px; }
        """

    # ── Playback ──
    DISPLAY_FPS = 30  # fixed screen refresh rate

    def _get_frame_skip(self):
        """Calculate how many frames to advance per display tick."""
        speed = float(self.combo_speed.currentText().replace("x", ""))
        # At 1x we want real-time = DATA_FPS frames per second
        # Display updates at DISPLAY_FPS, so skip = DATA_FPS * speed / DISPLAY_FPS
        return max(1, round(DATA_FPS * speed / self.DISPLAY_FPS))

    def _toggle_play(self):
        if self.is_playing:
            self.is_playing = False
            self.timer.stop()
            self.btn_play.setText("Play")
        else:
            self.is_playing = True
            self.btn_play.setText("Pause")
            self.timer.start(int(1000 / self.DISPLAY_FPS))

    def _speed_changed(self, text):
        pass  # frame skip is recalculated each tick, no timer change needed

    def _next_frame(self):
        skip = self._get_frame_skip()
        idx = self.current_frame_idx + skip
        if idx >= self.total_frames:
            idx = 0
        self.slider.blockSignals(True)
        self.slider.setValue(idx)
        self.slider.blockSignals(False)
        self._show_frame(idx)

    def _slider_changed(self, value):
        self._show_frame(value)

    def _toggle_trajectory(self, checked):
        self.show_trajectory = checked
        self.btn_trajectory.setText("Hide Trajectory" if checked else "Show Trajectory")
        self._show_frame(self.current_frame_idx)

    def _toggle_relative_trajectory(self, checked):
        self.show_relative_trajectory = checked
        self.btn_rel_trajectory.setText("Hide Relative Traj." if checked else "Show Relative Trajectory")
        self._show_frame(self.current_frame_idx)

    def _show_frame(self, idx):
        """Read and display a video frame with overlaid markers."""
        self.current_frame_idx = idx
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.cap.read()
        if not ret:
            return

        # Get marker positions for this frame (pixel space)
        marker_pos = {}
        trc_idx = min(idx, self.n_frames_px - 1)
        for m in self.marker_names:
            x = self.pos_px[m]["x"][trc_idx]
            y = self.pos_px[m]["y"][trc_idx]
            marker_pos[m] = (x, y)

        # Draw trajectory trail for selected marker
        if self.show_trajectory and self.selected_marker and self.selected_marker in self.pos_px:
            m = self.selected_marker
            color_hex = MARKER_COLORS.get(m, "#ffffff")
            cr = int(color_hex[1:3], 16)
            cg = int(color_hex[3:5], 16)
            cb = int(color_hex[5:7], 16)
            overlay = frame.copy()
            for i in range(1, trc_idx + 1):
                x1 = self.pos_px[m]["x"][i - 1]
                y1 = self.pos_px[m]["y"][i - 1]
                x2 = self.pos_px[m]["x"][i]
                y2 = self.pos_px[m]["y"][i]
                if x1 == 0 or y1 == 0 or x2 == 0 or y2 == 0:
                    continue
                alpha = 0.3 + 0.7 * (i / max(trc_idx, 1))
                cv2.line(overlay, (int(x1), int(y1)), (int(x2), int(y2)),
                         (int(cb * alpha), int(cg * alpha), int(cr * alpha)), 2)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Draw relative trajectory (relative to Hip)
        if self.show_relative_trajectory and self.selected_marker and self.selected_marker in self.pos_px and "Hip" in self.pos_px:
            m = self.selected_marker
            color_hex = MARKER_COLORS.get(m, "#ffffff")
            cr = int(color_hex[1:3], 16)
            cg = int(color_hex[3:5], 16)
            cb = int(color_hex[5:7], 16)
            # Current Hip position as anchor point for drawing
            hip_cx = self.pos_px["Hip"]["x"][trc_idx]
            hip_cy = self.pos_px["Hip"]["y"][trc_idx]
            overlay = frame.copy()
            for i in range(1, trc_idx + 1):
                # Relative position = marker - hip at each frame, drawn at current hip
                dx1 = self.pos_px[m]["x"][i-1] - self.pos_px["Hip"]["x"][i-1]
                dy1 = self.pos_px[m]["y"][i-1] - self.pos_px["Hip"]["y"][i-1]
                dx2 = self.pos_px[m]["x"][i] - self.pos_px["Hip"]["x"][i]
                dy2 = self.pos_px[m]["y"][i] - self.pos_px["Hip"]["y"][i]
                rx1, ry1 = hip_cx + dx1, hip_cy + dy1
                rx2, ry2 = hip_cx + dx2, hip_cy + dy2
                if rx1 == 0 or ry1 == 0 or rx2 == 0 or ry2 == 0:
                    continue
                alpha = 0.3 + 0.7 * (i / max(trc_idx, 1))
                cv2.line(overlay, (int(rx1), int(ry1)), (int(rx2), int(ry2)),
                         (int(cb * alpha), int(cg * alpha), int(cr * alpha)), 2)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Build angle info for the selected marker's joint
        angle_info = None
        if self.selected_marker:
            joint_name = MARKER_JOINT_MAP.get(self.selected_marker)
            mot_idx = min(idx, self.n_frames_mot - 1)
            if joint_name and joint_name in JOINT_ANGLE_MARKERS and joint_name in self.ang_kin:
                parent, vertex, child = JOINT_ANGLE_MARKERS[joint_name]
                if parent in marker_pos and vertex in marker_pos and child in marker_pos:
                    vx, vy = marker_pos[vertex]
                    px, py = marker_pos[parent]
                    cx, cy = marker_pos[child]
                    if not (vx == 0 or vy == 0 or px == 0 or py == 0 or cx == 0 or cy == 0):
                        ang1 = np.degrees(np.arctan2(-(py - vy), px - vx))
                        ang2 = np.degrees(np.arctan2(-(cy - vy), cx - vx))
                        start_a = min(ang1, ang2)
                        end_a = max(ang1, ang2)
                        if end_a - start_a > 180:
                            start_a, end_a = end_a, start_a + 360
                        arc_sweep = (end_a - start_a) % 360
                        angle_info = {
                            'joint_pos': (vx, vy),
                            'parent_pos': (px, py),
                            'child_pos': (cx, cy),
                            'display_val': arc_sweep,
                        }

        self.video_widget.set_frame(frame, marker_pos, self.marker_names, angle_info)

        # Update frame label
        time_val = self.time_m[min(idx, self.n_frames_m - 1)] if idx < self.n_frames_m else 0
        self.lbl_frame.setText(f"F {idx}/{self.total_frames-1}  |  {time_val:.3f}s")

        # Update kinematics data if a marker is selected
        if self.selected_marker:
            self._update_kinematics_display(idx)

        # Update graph vline
        if self.graph_marker and self.graph_widget.axes:
            t = self.time_m[min(idx, self.n_frames_m - 1)]
            self.graph_widget.update_vline(t)

    def _on_marker_selected(self, marker_name):
        """Called when user clicks a marker."""
        self.selected_marker = marker_name
        self.lbl_selected.setText(f"Selected: {marker_name}")
        self.lbl_selected.setStyleSheet("color: #89b4fa; font-size: 12px; font-weight: bold; padding: 2px;")
        self.btn_trajectory.setEnabled(True)
        self.btn_rel_trajectory.setEnabled(True)
        self._update_kinematics_display(self.current_frame_idx)
        self._generate_graph()
        # Re-render current frame to show selection highlight
        self._show_frame(self.current_frame_idx)

    def _update_kinematics_display(self, idx):
        """Update the kinematics values for the selected marker at frame idx."""
        m = self.selected_marker
        if not m:
            return

        m_idx = min(idx, self.n_frames_m - 1)

        # Linear kinematics
        if m in self.kin:
            k = self.kin[m]
            self.data_values[0].setText(f"({k['x'][m_idx]:.4f}, {k['y'][m_idx]:.4f})")
            self.data_values[1].setText(f"{k['speed'][m_idx]:.4f}")
            self.data_values[2].setText(f"{k['accel'][m_idx]:.4f}")
        else:
            for i in range(3):
                self.data_values[i].setText("N/A")

        # Angular kinematics
        joint = MARKER_JOINT_MAP.get(m)
        mot_idx = min(idx, self.n_frames_mot - 1)
        if joint and joint in self.ang_kin:
            ak = self.ang_kin[joint]
            self.data_values[3].setText(f"{ak['angle'][mot_idx]:.2f}")
            self.data_values[4].setText(f"{ak['omega'][mot_idx]:.2f}")
            self.data_values[5].setText(f"{ak['alpha'][mot_idx]:.2f}")
        else:
            for i in range(3, 6):
                self.data_values[i].setText("N/A")

    def _generate_graph(self):
        """Generate kinematics graph for the selected marker."""
        m = self.selected_marker
        if not m or m not in self.kin:
            return

        self.graph_marker = m
        joint = MARKER_JOINT_MAP.get(m)
        self.graph_widget.plot_kinematics(
            m, self.time_m, self.kin[m],
            joint, self.ang_kin, self.time_mot,
            min(self.current_frame_idx, self.n_frames_m - 1)
        )

    def closeEvent(self, event):
        self.timer.stop()
        if self.cap:
            self.cap.release()
        event.accept()


# ──────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────
def main():
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))
    window = BiomechanicsGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
