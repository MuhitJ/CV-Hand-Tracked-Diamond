import cv2
import numpy as np
import math
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_PATH = "/Applications/CSC 242/hand_diamond/hand_landmarker.task"

TOP    = np.array([0, 2, 0], dtype=float)
BOTTOM = np.array([0, -2, 0], dtype=float)
MID_UP = np.array([
    [1.2, 0.5, 0], [0, 0.5, 1.2], [-1.2, 0.5, 0], [0, 0.5, -1.2],
    [0.85, 0.5, 0.85], [-0.85, 0.5, 0.85], [-0.85, 0.5, -0.85], [0.85, 0.5, -0.85]
], dtype=float)
MID_DOWN = np.array([
    [0.8, -0.5, 0], [0, -0.5, 0.8], [-0.8, -0.5, 0], [0, -0.5, -0.8]
], dtype=float)

T1 = (175, 216, 10)
T2 = (155, 200, 8)
T3 = (130, 180, 6)
T4 = (100, 150, 4)

FACES = [
    (TOP, MID_UP[0], MID_UP[4], T1),
    (TOP, MID_UP[4], MID_UP[1], T1),
    (TOP, MID_UP[1], MID_UP[5], T1),
    (TOP, MID_UP[5], MID_UP[2], T1),
    (TOP, MID_UP[2], MID_UP[6], T1),
    (TOP, MID_UP[6], MID_UP[3], T1),
    (TOP, MID_UP[3], MID_UP[7], T1),
    (TOP, MID_UP[7], MID_UP[0], T1),
    (MID_UP[0], MID_DOWN[0], MID_UP[4], T2),
    (MID_UP[4], MID_DOWN[1], MID_UP[1], T2),
    (MID_UP[1], MID_DOWN[1], MID_UP[5], T2),
    (MID_UP[5], MID_DOWN[2], MID_UP[2], T2),
    (MID_UP[2], MID_DOWN[2], MID_UP[6], T2),
    (MID_UP[6], MID_DOWN[3], MID_UP[3], T2),
    (MID_UP[3], MID_DOWN[3], MID_UP[7], T2),
    (MID_UP[7], MID_DOWN[0], MID_UP[0], T2),
    (MID_DOWN[0], BOTTOM, MID_DOWN[1], T3),
    (MID_DOWN[1], BOTTOM, MID_DOWN[2], T3),
    (MID_DOWN[2], BOTTOM, MID_DOWN[3], T4),
    (MID_DOWN[3], BOTTOM, MID_DOWN[0], T4),
]

def rotate_x(v, a):
    c, s = math.cos(a), math.sin(a)
    return v @ np.array([[1,0,0],[0,c,-s],[0,s,c]]).T

def rotate_y(v, a):
    c, s = math.cos(a), math.sin(a)
    return v @ np.array([[c,0,s],[0,1,0],[-s,0,c]]).T

def project(v, W, H, sc=1.0, fov=5):
    z = v[2] + fov
    if abs(z) < 0.001: z = 0.001
    return (int((v[0]/z)*W*0.25*sc + W//2), int((-v[1]/z)*H*0.25*sc + H//2))

latest_result = None
def result_callback(result, output_image, timestamp_ms):
    global latest_result
    latest_result = result

base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.LIVE_STREAM,
    num_hands=1,
    result_callback=result_callback
)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
W, H = 800, 600
angle_x, angle_y = 0.2, 0.2
prev_x, prev_y = None, None
timestamp = 0
scale = 1.0
frozen = False
auto_spin = False
fist_prev = False

print("Tiffany Diamond running! Press Q to quit.")
print("Gestures: Index finger=rotate | Open palm=auto-spin | Fist=freeze | Pinch=scale")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (W, H))
    webcam_small = cv2.resize(frame, (200, 150))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    frame[H-160:H-10, W-210:W-10] = webcam_small

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    detector.detect_async(mp_image, timestamp)
    timestamp += 33

    status = "NO HAND"

    if latest_result and latest_result.hand_landmarks:
        lm = latest_result.hand_landmarks[0]

        thumb_tip  = lm[4]
        index_tip  = lm[8]
        middle_tip = lm[12]
        ring_tip   = lm[16]
        pinky_tip  = lm[20]

        index_up  = index_tip.y  < lm[6].y
        middle_up = middle_tip.y < lm[10].y
        ring_up   = ring_tip.y   < lm[14].y
        pinky_up  = pinky_tip.y  < lm[18].y
        thumb_up  = thumb_tip.x  < lm[3].x

        fingers_up = sum([index_up, middle_up, ring_up, pinky_up])
        pinch_dist = ((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)**0.5

        # FIST = toggle freeze
        is_fist = fingers_up == 0
        if is_fist and not fist_prev:
            frozen = not frozen
        fist_prev = is_fist

        # OPEN PALM = auto spin
        if fingers_up == 4 and thumb_up:
            auto_spin = True
            status = "AUTO-SPIN 🖐"
        else:
            auto_spin = False

        # PINCH = scale
        if pinch_dist < 0.05:
            scale = min(scale + 0.02, 2.5)
            status = "SCALE UP 🤏"
        elif index_up and middle_up and not ring_up and not pinky_up:
            scale = max(scale - 0.02, 0.3)
            status = "SCALE DOWN ✌️"


        # INDEX ONLY = rotate
                # INDEX ONLY = rotate
        if index_up and not middle_up and not is_fist:
            hand_x, hand_y = index_tip.x, index_tip.y
            if not frozen and prev_x is not None:
                dx = hand_x - prev_x
                dy = hand_y - prev_y
                angle_y += dx * 6
                angle_x += dy * 6
                spin_velocity_y = dx * 6
                spin_velocity_x = dy * 6
            prev_x, prev_y = hand_x, hand_y
            if not frozen:
                status = "ROTATING ☝️"
        else:
            prev_x, prev_y = None, None


        if frozen:
            status = "FROZEN ✊"

        fx = int(index_tip.x * W)
        fy = int(index_tip.y * H)
        cv2.circle(frame, (fx, fy), 10, (175, 216, 10), -1)
        cv2.circle(frame, (fx, fy), 12, (255, 255, 255), 1)
    else:
        prev_x, prev_y = None, None
        fist_prev = False

    # Auto spin
    if auto_spin:
        angle_y += 0.03

    # Draw diamond
    depth_faces = []
    for v0, v1, v2, color in FACES:
        rv0 = rotate_y(rotate_x(v0, angle_x), angle_y)
        rv1 = rotate_y(rotate_x(v1, angle_x), angle_y)
        rv2 = rotate_y(rotate_x(v2, angle_x), angle_y)
        depth = (rv0[2] + rv1[2] + rv2[2]) / 3
        depth_faces.append((depth, rv0, rv1, rv2, color))

    depth_faces.sort(key=lambda f: f[0])
    overlay = frame.copy()

    for _, rv0, rv1, rv2, color in depth_faces:
        pts = np.array([project(rv0,W,H,scale), project(rv1,W,H,scale), project(rv2,W,H,scale)], np.int32)
        cv2.fillPoly(overlay, [pts], color)
        cv2.polylines(overlay, [pts], True, (220, 240, 200), 1)

    cv2.addWeighted(overlay, 0.9, frame, 0.1, 0, frame)

    cv2.putText(frame, "Tiffany Diamond", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (175, 216, 10), 2)
    cv2.putText(frame, f"Mode: {status}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    cv2.putText(frame, f"Scale: {scale:.1f}x", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    cv2.putText(frame, "Index=rotate | Palm=spin | Fist=freeze | Pinch=scale", (10, H-20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)

    cv2.imshow("Tiffany Diamond", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
detector.close()
