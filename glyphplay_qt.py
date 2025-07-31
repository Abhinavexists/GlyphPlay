import sys
import cv2
import numpy as np
import mediapipe as mp
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QFrame)
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal, QPointF
from PyQt6.QtGui import QPixmap, QPainter, QPen, QBrush, QColor, QFont, QPainterPath
from font_data import font_data
from scipy.interpolate import splprep, splev
from perlin_noise import PerlinNoise

class CameraThread(QThread):
    frame_ready = pyqtSignal(np.ndarray)
    hand_data = pyqtSignal(dict)
    
    def __init__(self):
        super().__init__()
        self.running = True
        
    def run(self):
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
            model_complexity=1
        )
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            cap = cv2.VideoCapture(1)
        
        while self.running:
            ret, frame = cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                
                results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                
                hand_info = {
                    'landmarks': None,
                    'one_finger': False,
                    'v_sign': False,
                    'three_fingers': False,
                    'all_fingers': False
                }
                
                if results.multi_hand_landmarks:
                    hand_landmarks = results.multi_hand_landmarks[0]
                    landmarks = hand_landmarks.landmark
                    
                    thumb_tip = landmarks[4]
                    index_tip = landmarks[8]
                    middle_tip = landmarks[12]
                    ring_tip = landmarks[16]
                    pinky_tip = landmarks[20]
                    
                    index_extended = index_tip.y < landmarks[6].y
                    middle_extended = middle_tip.y < landmarks[10].y
                    ring_closed = ring_tip.y > landmarks[14].y
                    pinky_closed = pinky_tip.y > landmarks[18].y
                    thumb_closed = thumb_tip.y > landmarks[2].y
                    
                    one_finger = index_extended and not middle_extended and ring_closed and pinky_closed and thumb_closed
                    v_sign = index_extended and middle_extended and ring_closed and pinky_closed and thumb_closed
                    three_fingers = index_extended and middle_extended and (ring_tip.y < landmarks[14].y) and pinky_closed and thumb_closed
                    all_fingers = index_extended and middle_extended and (ring_tip.y < landmarks[14].y) and (pinky_tip.y < landmarks[18].y) and (thumb_tip.y < landmarks[2].y)
                    
                    hand_info.update({
                        'landmarks': landmarks,
                        'one_finger': one_finger,
                        'v_sign': v_sign,
                        'three_fingers': three_fingers,
                        'all_fingers': all_fingers
                    })
                
                self.frame_ready.emit(frame)
                self.hand_data.emit(hand_info)
        
        cap.release()
        hands.close()

class GlyphRenderer(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(600, 400)
        self.current_char = 'A'
        self.distortion_value = 0.0
        self.smoothness_value = 0
        self.time_anim = 0
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Clear background
        painter.fillRect(self.rect(), QColor(0, 0, 0))
        
        # Get glyph data
        base_strokes = font_data.get(self.current_char, [])
        if base_strokes:
            # Apply smoothing
            processed_strokes = self.apply_spline_smoothing(base_strokes) if self.smoothness_value > 0 else base_strokes
            # Apply distortion
            distorted_strokes = self.apply_animated_distortion(processed_strokes, self.distortion_value, self.time_anim)
            
            # Draw glyph
            self.draw_glyph(painter, distorted_strokes)
        
        # Draw alphabet ring
        self.draw_alphabet_ring(painter)
        
    def draw_glyph(self, painter, strokes):
        if not strokes:
            return
            
        # Set up pen for glyph
        pen = QPen(QColor(255, 255, 255), 8, Qt.PenStyle.SolidLine)
        painter.setPen(pen)
        
        # Calculate center and scale
        center_x = self.width() // 2
        center_y = self.height() // 2
        scale = min(self.width(), self.height()) * 0.3
        
        for stroke in strokes:
            if len(stroke) == 0:
                continue
                
            # Ensure stroke is 2D
            if stroke.ndim == 1:
                stroke = stroke.reshape(-1, 2)
            
            # Create path for stroke
            path = QPainterPath()
            
            for i, point in enumerate(stroke):
                x = point[0] * scale + center_x
                y = -point[1] * scale + center_y
                
                if i == 0:
                    path.moveTo(x, y)
                else:
                    path.lineTo(x, y)
            
            painter.drawPath(path)
    
    def draw_alphabet_ring(self, painter):
        center_x = self.width() // 2
        center_y = self.height() // 2
        radius = 150
        
        # Set up font
        font = QFont("Arial", 12)
        painter.setFont(font)
        
        for i, letter in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
            angle = (i / 26) * 2 * np.pi - np.pi/2
            x = center_x + radius * np.cos(angle)
            y = center_y + radius * np.sin(angle)
            
            if letter == self.current_char:
                # Highlight current character
                painter.setBrush(QBrush(QColor(0, 255, 150)))
                painter.setPen(QPen(QColor(0, 255, 150), 2))
                painter.drawEllipse(QPointF(x, y), 20, 20)
                painter.setPen(QPen(QColor(0, 0, 0), 1))
            else:
                # Regular character
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.setPen(QPen(QColor(255, 255, 255), 1))
                painter.drawEllipse(QPointF(x, y), 15, 15)
                painter.setPen(QPen(QColor(255, 255, 255), 1))
            
            # Draw letter
            painter.drawText(QPointF(x - 6, y + 4), letter)
    
    def apply_spline_smoothing(self, strokes, num_points=100):
        smoothed_strokes = []
        for stroke in strokes:
            if len(stroke) < 3:
                smoothed_strokes.append(stroke)
                continue
                
            if stroke.ndim == 1:
                stroke = stroke.reshape(-1, 2)
            
            x, y = stroke[:, 0], stroke[:, 1]
            try:
                tck, u = splprep([x, y], s=0, per=False)
                u_new = np.linspace(u.min(), u.max(), num_points)
                x_new, y_new = splev(u_new, tck, der=0)
                smoothed_strokes.append(np.array([x_new, y_new]).T)
            except:
                smoothed_strokes.append(stroke)
        
        return smoothed_strokes
    
    def apply_animated_distortion(self, strokes, factor, time):
        if factor == 0:
            return strokes
            
        noise = PerlinNoise(octaves=3, seed=1)
        distorted_strokes = []
        
        for stroke in strokes:
            if len(stroke) == 0:
                distorted_strokes.append(stroke)
                continue
                
            if stroke.ndim == 1:
                stroke = stroke.reshape(-1, 2)
            
            distorted_stroke = np.copy(stroke).astype(np.float32)
            
            for i, point in enumerate(distorted_stroke):
                # Add Perlin noise distortion
                noise_x = noise([point[0] * 0.1, time * 0.1]) * factor
                noise_y = noise([point[1] * 0.1, time * 0.1 + 100]) * factor
                
                distorted_stroke[i, 0] += noise_x
                distorted_stroke[i, 1] += noise_y
            
            distorted_strokes.append(distorted_stroke)
        
        return distorted_strokes
    
    def update_glyph(self, char, distortion, smoothness):
        self.current_char = char
        self.distortion_value = distortion
        self.smoothness_value = smoothness
        self.time_anim += 0.01
        self.update()

class GlyphPlayUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GlyphPlay - Gestural Font Editor")
        self.setGeometry(100, 100, 1400, 900)
        
        # Initialize variables
        self.alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        self.current_char = 'A'
        self.distortion_value = 0.0
        self.smoothness_value = 0
        self.current_mode = "CHARACTER"
        
        # Setup UI
        self.setup_ui()
        
        # Setup camera thread
        self.camera_thread = CameraThread()
        self.camera_thread.frame_ready.connect(self.update_camera_feed)
        self.camera_thread.hand_data.connect(self.process_hand_data)
        self.camera_thread.start()
        
        # Setup animation timer
        self.anim_timer = QTimer()
        self.anim_timer.timeout.connect(self.update_animation)
        self.anim_timer.start(16)  # ~60 FPS
        
    def setup_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        layout = QHBoxLayout(main_widget)
        
        left_panel = QVBoxLayout()
        
        title_label = QLabel("GLYPHPLAY")
        title_font = QFont("Arial", 24, QFont.Weight.Bold)
        title_label.setFont(title_font)
        title_label.setStyleSheet("color: #FFA500; margin: 10px;")
        left_panel.addWidget(title_label)
        
        subtitle_label = QLabel("gestural font editor ?")
        subtitle_font = QFont("Arial", 12)
        subtitle_label.setFont(subtitle_font)
        subtitle_label.setStyleSheet("color: white; margin: 5px;")
        left_panel.addWidget(subtitle_label)
        
        self.glyph_renderer = GlyphRenderer()
        left_panel.addWidget(self.glyph_renderer)
        
        self.mode_label = QLabel("ADJUSTING: CHARACTER")
        self.mode_label.setStyleSheet("""
            QLabel {
                background-color: #006400;
                color: white;
                padding: 10px;
                border: 2px solid white;
                border-radius: 5px;
                font-size: 14px;
                font-weight: bold;
            }
        """)
        left_panel.addWidget(self.mode_label)
        
        layout.addLayout(left_panel)
        
        # Right panel (parameters and camera)
        right_panel = QVBoxLayout()
        
        # Parameter panel
        param_frame = QFrame()
        param_frame.setStyleSheet("""
            QFrame {
                background-color: black;
                border: 2px solid #FFB6C1;
                border-radius: 5px;
                padding: 10px;
            }
        """)
        param_frame.setFixedSize(350, 200)
        
        param_layout = QVBoxLayout(param_frame)
        
        # Character parameter
        char_layout = QHBoxLayout()
        char_label = QLabel("CHARACTER:")
        char_label.setStyleSheet("color: white; font-size: 12px;")
        self.char_value = QLabel("A")
        self.char_value.setStyleSheet("color: #00FF96; font-size: 14px; font-weight: bold;")
        char_layout.addWidget(char_label)
        char_layout.addWidget(self.char_value)
        char_layout.addStretch()
        param_layout.addLayout(char_layout)
        
        dist_layout = QHBoxLayout()
        dist_label = QLabel("DISTORTION:")
        dist_label.setStyleSheet("color: white; font-size: 12px;")
        self.dist_value = QLabel("0.00")
        self.dist_value.setStyleSheet("color: #FF9600; font-size: 14px; font-weight: bold;")
        dist_layout.addWidget(dist_label)
        dist_layout.addWidget(self.dist_value)
        dist_layout.addStretch()
        param_layout.addLayout(dist_layout)
        
        smooth_layout = QHBoxLayout()
        smooth_label = QLabel("SMOOTHNESS:")
        smooth_label.setStyleSheet("color: white; font-size: 12px;")
        self.smooth_value = QLabel("0")
        self.smooth_value.setStyleSheet("color: #9600FF; font-size: 14px; font-weight: bold;")
        smooth_layout.addWidget(smooth_label)
        smooth_layout.addWidget(self.smooth_value)
        smooth_layout.addStretch()
        param_layout.addLayout(smooth_layout)
        
        right_panel.addWidget(param_frame)
        
        self.camera_label = QLabel("CAMERA")
        self.camera_label.setStyleSheet("color: white; font-size: 10px; margin: 5px;")
        right_panel.addWidget(self.camera_label)
        
        self.camera_display = QLabel()
        self.camera_display.setStyleSheet("border: 2px solid #FFB6C1; border-radius: 5px;")
        self.camera_display.setFixedSize(320, 240)
        right_panel.addWidget(self.camera_display)
        
        right_panel.addStretch()
        layout.addLayout(right_panel)
        
        self.setStyleSheet("background-color: black;")
        
    def update_camera_feed(self, frame):
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_image = frame.data.tobytes()
        
        from PyQt6.QtGui import QImage
        q_image = QImage(q_image, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        
        scaled_pixmap = pixmap.scaled(320, 240, Qt.AspectRatioMode.KeepAspectRatio)
        self.camera_display.setPixmap(scaled_pixmap)
        
    def process_hand_data(self, hand_info):
        if not hand_info['landmarks']:
            return
            
        landmarks = hand_info['landmarks']
        
        if hand_info['one_finger']:
            index_tip = landmarks[8]
            wrist = landmarks[0]
            angle = np.degrees(np.arctan2(index_tip.y - wrist.y, index_tip.x - wrist.x))
            char_index = int(np.interp(angle, [-180, 180], [0, len(self.alphabet) - 1]))
            char_index = max(0, min(char_index, len(self.alphabet) - 1))
            self.current_char = self.alphabet[char_index]
            self.current_mode = "CHARACTER"
            
        elif hand_info['v_sign']:
            index_tip = landmarks[8]
            middle_tip = landmarks[12]
            angle = np.degrees(np.arctan2(middle_tip.y - index_tip.y, middle_tip.x - index_tip.x))
            self.distortion_value = np.interp(angle, [-90, 90], [0, 0.4])
            self.current_mode = "DISTORTION"
            
        elif hand_info['three_fingers']:
            index_tip = landmarks[8]
            middle_tip = landmarks[12]
            angle = np.degrees(np.arctan2(middle_tip.y - index_tip.y, middle_tip.x - index_tip.x))
            self.smoothness_value = int(np.interp(angle, [-90, 90], [0, 6]))
            self.current_mode = "SMOOTHNESS"
        
        self.update_ui()
        
    def update_ui(self):
        self.char_value.setText(self.current_char)
        self.dist_value.setText(f"{self.distortion_value:.2f}")
        self.smooth_value.setText(str(self.smoothness_value))
        
        self.mode_label.setText(f"ADJUSTING: {self.current_mode}")
        
        self.glyph_renderer.update_glyph(self.current_char, self.distortion_value, self.smoothness_value)
        
    def update_animation(self):
        self.glyph_renderer.time_anim += 0.01
        self.glyph_renderer.update()
        
    def closeEvent(self, event):
        self.camera_thread.running = False
        self.camera_thread.wait()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = GlyphPlayUI()
    window.show()
    sys.exit(app.exec()) 