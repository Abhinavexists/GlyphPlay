import cv2
import mediapipe as mp
import numpy as np
import math
from font_data import font_data
from scipy.interpolate import splprep, splev
from perlin_noise import PerlinNoise

noise = PerlinNoise(octaves=4, seed=1)
def apply_animated_distortion(strokes, factor, time):
    if factor == 0 or strokes is None: return strokes
    
    distorted_strokes = []
    for stroke in strokes:
        if len(stroke) == 0: 
            distorted_strokes.append(stroke)
            continue
            
        distorted_points = np.copy(stroke)
        for i in range(len(distorted_points)):
            x, y = distorted_points[i]
            noise_val = noise([x*2, y*2, time])
            offset = factor * noise_val
            distorted_points[i] = (x + offset, y + offset)
        distorted_strokes.append(distorted_points)
    return distorted_strokes

def apply_spline_smoothing(strokes, num_points=100):
    if strokes is None: return None
    
    smoothed_strokes = []
    for stroke in strokes:
        if len(stroke) < 4:
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

def draw_glyph(image, strokes, scale=400, offset=(150, 500)):
    if strokes is None: return
    
    for i, stroke in enumerate(strokes):
        if len(stroke) == 0: continue
        if stroke.ndim == 1:
            stroke = stroke.reshape(-1, 2)
        scaled_points = np.copy(stroke).astype(np.float32)
        scaled_points[:, 0] = scaled_points[:, 0] * scale + offset[0]
        scaled_points[:, 1] = scaled_points[:, 1] * scale * -1 + offset[1]
        
        points = scaled_points.astype(np.int32)
        
        for thickness in [12, 10]:
            cv2.polylines(image, [points], isClosed=False, color=(50, 100, 150), thickness=thickness)
        
        cv2.polylines(image, [points], isClosed=False, color=(255, 255, 255), thickness=8)

def draw_ui(image, w, h, mode, char, dist_val, smooth_val):
    title = "GLYPHPLAY"
    for i in range(4, 0, -1):
        cv2.putText(image, title, (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 
                   (int(255/i), int(165/i), int(0/i)), i)
    
    subtitle = "gestural font editor ?"
    cv2.putText(image, subtitle, (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    panel_x = w - 400
    panel_y = 50
    panel_w = 350
    panel_h = 200
    cv2.rectangle(image, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), 
                 (255, 192, 203), 2)  # Pink border
    cv2.rectangle(image, (panel_x + 2, panel_y + 2), (panel_x + panel_w - 2, panel_y + panel_h - 2), 
                 (0, 0, 0), -1)  # Black background
    
    # Parameter labels and values (white text)
    param_y = panel_y + 30
    param_spacing = 40
    
    # Character parameter
    cv2.putText(image, "CHARACTER:", (panel_x + 20, param_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
               (255, 255, 255), 1)
    cv2.putText(image, f"{char}", (panel_x + 200, param_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 
               (0, 255, 150), 2)
    
    # Distortion parameter
    cv2.putText(image, "DISTORTION:", (panel_x + 20, param_y + param_spacing), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
               (255, 255, 255), 1)
    cv2.putText(image, f"{dist_val:.2f}", (panel_x + 200, param_y + param_spacing), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 
               (255, 150, 0), 2)
    
    # Smoothness parameter
    cv2.putText(image, "SMOOTHNESS:", (panel_x + 20, param_y + param_spacing * 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
               (255, 255, 255), 1)
    cv2.putText(image, f"{smooth_val}", (panel_x + 200, param_y + param_spacing * 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 
               (150, 0, 255), 2)
    
    # Current mode indicator (bottom) - dark green bar like in reference
    mode_bg_x = 50
    mode_bg_y = h - 80
    mode_bg_w = 300
    mode_bg_h = 50
    
    # Mode background - dark green like in reference
    cv2.rectangle(image, (mode_bg_x, mode_bg_y), (mode_bg_x + mode_bg_w, mode_bg_y + mode_bg_h), 
                 (0, 100, 0), -1)  # Dark green
    cv2.rectangle(image, (mode_bg_x, mode_bg_y), (mode_bg_x + mode_bg_w, mode_bg_y + mode_bg_h), 
                 (255, 255, 255), 2)  # White border
    
    # Mode text
    cv2.putText(image, f"ADJUSTING: {mode}", (mode_bg_x + 20, mode_bg_y + 35), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Alphabet ring visualization (circular arrangement) - white outlined letters
    center_x = w // 2
    center_y = h // 2
    radius = 200
    
    # Draw alphabet ring with white outlines like in reference
    for i, letter in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
        angle = (i / 26) * 2 * np.pi - np.pi/2  # Start from top
        x = int(center_x + radius * np.cos(angle))
        y = int(center_y + radius * np.sin(angle))
        
        # Highlight current character with bright green circle like in reference
        if letter == char:
            cv2.circle(image, (x, y), 25, (0, 255, 150), -1)  # Bright green fill
            cv2.putText(image, letter, (x-8, y+8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        else:
            # White outlined circles for other letters
            cv2.circle(image, (x, y), 20, (255, 255, 255), 1)  # White outline
            cv2.putText(image, letter, (x-6, y+6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)


# --- INITIALIZATION & MAIN LOOP ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1, 
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
    model_complexity=1
)
mp_draw = mp.solutions.drawing_utils

# Check if camera opened successfully
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera at index 0. Trying index 1...")
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Error: Could not open camera at index 1 either.")
        exit(1)
    else:
        print("Successfully opened camera at index 1")
else:
    print("Successfully opened camera at index 0")

# State variables
MODES = ["CHARACTER", "DISTORTION", "SMOOTHNESS"]
mode_index = 0
current_mode = MODES[mode_index]
alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
current_char = 'A'
distortion_value = 0.0
smoothness_value = 0
time_anim = 0

# Gesture detection variables
mode_switch_cooldown = 0
last_mode = current_mode

while cap.isOpened():
    success, image = cap.read()
    if not success: break

    time_anim += 0.01
    if mode_switch_cooldown > 0: 
        mode_switch_cooldown -= 1

    # Flip image for mirror effect
    image = cv2.flip(image, 1)
    h, w, _ = image.shape
    
    # Create canvas with pure black background like in reference
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[:] = (0, 0, 0)  # Pure black background
    
    # Process hand landmarks
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # Draw hand landmarks for debugging
        mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        landmarks = hand_landmarks.landmark
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        pinky_tip = landmarks[20]
        
        # Calculate distances
        dist_thumb_index = np.hypot(thumb_tip.x - index_tip.x, thumb_tip.y - index_tip.y)
        dist_thumb_middle = np.hypot(thumb_tip.x - middle_tip.x, thumb_tip.y - middle_tip.y)
        dist_thumb_ring = np.hypot(thumb_tip.x - ring_tip.x, thumb_tip.y - ring_tip.y)
        dist_thumb_pinky = np.hypot(thumb_tip.x - pinky_tip.x, thumb_tip.y - pinky_tip.y)
        
        # Improved thresholds
        pinch_threshold = 0.08  # Increased from 0.06
        multi_pinch_threshold = 0.12
        
        # Draw finger tip circles for visual feedback
        for i, (lm, color) in enumerate([(thumb_tip, (0, 255, 255)), (index_tip, (255, 0, 255)), 
                                       (middle_tip, (255, 255, 0)), (ring_tip, (0, 255, 0)), 
                                       (pinky_tip, (255, 0, 0))]):
            cv2.circle(image, (int(lm.x * w), int(lm.y * h)), 15, color, -1)
            cv2.putText(image, str(i+1), (int(lm.x * w) + 20, int(lm.y * h)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Improved gesture detection logic using V-sign and finger positions
        # Check if index and middle fingers are extended (V-sign)
        index_extended = index_tip.y < landmarks[6].y  # Index tip above middle joint
        middle_extended = middle_tip.y < landmarks[10].y  # Middle tip above middle joint
        ring_closed = ring_tip.y > landmarks[14].y  # Ring tip below middle joint
        pinky_closed = pinky_tip.y > landmarks[18].y  # Pinky tip below middle joint
        thumb_closed = thumb_tip.y > landmarks[2].y  # Thumb tip below base joint
        
        # One finger extended (just index)
        one_finger = index_extended and not middle_extended and ring_closed and pinky_closed and thumb_closed
        
        # V-sign gesture (index + middle extended, others closed)
        v_sign = index_extended and middle_extended and ring_closed and pinky_closed and thumb_closed
        
        # Three fingers extended (index + middle + ring)
        three_fingers = index_extended and middle_extended and (ring_tip.y < landmarks[14].y) and pinky_closed and thumb_closed
        
        # All fingers extended
        all_fingers = index_extended and middle_extended and (ring_tip.y < landmarks[14].y) and (pinky_tip.y < landmarks[18].y) and (thumb_tip.y < landmarks[2].y)
        
        # Control values based on gesture type
        if one_finger:
            # One finger - change character by rotating hand
            angle = np.degrees(np.arctan2(index_tip.y - landmarks[0].y, index_tip.x - landmarks[0].x))
            char_index = int(np.interp(angle, [-180, 180], [0, len(alphabet) - 1]))
            char_index = max(0, min(char_index, len(alphabet) - 1))  # Clamp values
            current_char = alphabet[char_index]
            current_mode = "CHARACTER"
            
        elif v_sign:
            # Two fingers (V-sign) - change distortion by rotating hand
            angle = np.degrees(np.arctan2(middle_tip.y - index_tip.y, middle_tip.x - index_tip.x))
            distortion_value = np.interp(angle, [-90, 90], [0, 0.4])
            current_mode = "DISTORTION"
            
        elif three_fingers:
            # Three fingers - change smoothness by rotating hand
            angle = np.degrees(np.arctan2(middle_tip.y - index_tip.y, middle_tip.x - index_tip.x))
            smoothness_value = int(np.interp(angle, [-90, 90], [0, 6]))
            current_mode = "SMOOTHNESS"
        
        # Draw gesture indicators
        if one_finger:
            # Draw circle on index fingertip for one finger
            cv2.circle(image, (int(index_tip.x * w), int(index_tip.y * h)), 15, (255, 255, 0), -1)
            cv2.putText(image, "CHARACTER", (int(index_tip.x * w) + 20, int(index_tip.y * h)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
        elif v_sign:
            # Draw line between index and middle fingertips for V-sign
            cv2.line(image, (int(index_tip.x * w), int(index_tip.y * h)), 
                    (int(middle_tip.x * w), int(middle_tip.y * h)), (0, 255, 0), 3)
            # Draw circle at midpoint
            mid_x = int((index_tip.x + middle_tip.x) * w / 2)
            mid_y = int((index_tip.y + middle_tip.y) * h / 2)
            cv2.circle(image, (mid_x, mid_y), 10, (0, 255, 0), -1)
            cv2.putText(image, "DISTORTION", (mid_x + 20, mid_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
        elif three_fingers:
            # Draw line between index and middle fingertips for three fingers
            cv2.line(image, (int(index_tip.x * w), int(index_tip.y * h)), 
                    (int(middle_tip.x * w), int(middle_tip.y * h)), (0, 255, 255), 3)
            # Draw circle at midpoint
            mid_x = int((index_tip.x + middle_tip.x) * w / 2)
            mid_y = int((index_tip.y + middle_tip.y) * h / 2)
            cv2.circle(image, (mid_x, mid_y), 10, (0, 255, 255), -1)
            cv2.putText(image, "SMOOTHNESS", (mid_x + 20, mid_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Process and render glyph
    base_strokes = font_data.get(current_char)
    if base_strokes is None:
        print(f"Warning: No font data for character '{current_char}'")
        base_strokes = []
    
    processed_strokes = apply_spline_smoothing(base_strokes) if smoothness_value > 0 else base_strokes
    distorted_strokes = apply_animated_distortion(processed_strokes, distortion_value, time_anim)
    
    # Debug: Print stroke info (only when character changes)
    if len(distorted_strokes) > 0 and (not hasattr(draw_ui, 'last_char') or draw_ui.last_char != current_char):
        print(f"Rendering character '{current_char}' with {len(distorted_strokes)} strokes")
        draw_ui.last_char = current_char
    
    draw_glyph(canvas, distorted_strokes)
    draw_ui(canvas, w, h, current_mode, current_char, distortion_value, smoothness_value)

    # Show camera feed in corner with frame (matching reference design)
    cam_small = cv2.resize(image, (w // 4, h // 4))
    cam_h, cam_w, _ = cam_small.shape
    cam_x = w - cam_w - 30
    cam_y = 30
    
    # Add frame around camera feed with pink border
    cv2.rectangle(canvas, (cam_x - 5, cam_y - 5), (cam_x + cam_w + 5, cam_y + cam_h + 5), 
                 (255, 192, 203), 2)  # Pink border
    
    # Add camera feed directly
    canvas[cam_y:cam_y+cam_h, cam_x:cam_x+cam_w] = cam_small
    
    # Add "CAMERA" label in white
    cv2.putText(canvas, "CAMERA", (cam_x, cam_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
               (255, 255, 255), 1)
    
    cv2.imshow('GlyphPlay - Gestural Font Editor', canvas)
    if cv2.waitKey(5) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()