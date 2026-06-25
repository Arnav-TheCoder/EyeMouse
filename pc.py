import cv2
import time
import numpy as np
import pyautogui
import os
import mediapipe as mp
import requests
from collections import deque
import tkinter as tk
from threading import Thread

#SETUP & MODEL DOWNLOAD
MODEL_PATH = os.path.join("models", "face_landmarker.task")
if not os.path.exists("models"): os.makedirs("models")
if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    r = requests.get("https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task")
    with open(MODEL_PATH, "wb") as f: f.write(r.content)


#STATIC CENTER OVERLAY
overlay_color = "red"
root = None
canvas = None
dot = None

def flash_click_signal():
    """Turns the dot green briefly to signal a click."""
    if canvas and dot:
        canvas.itemconfig(dot, fill="#00FF00") # Bright Green
        try:
            root.after(200, lambda: canvas.itemconfig(dot, fill="red"))
        except: pass

def create_static_overlay():
    global root, canvas, dot
    root = tk.Tk()
    root.overrideredirect(True)
    root.wm_attributes("-topmost", True)
    
    try:
        root.wm_attributes("-transparentcolor", "white")
        root.config(bg='white')
    except:
        pass 
    
    sw = root.winfo_screenwidth()
    sh = root.winfo_screenheight()
    
    w, h = 50, 50
    x = (sw // 2) - (w // 2)
    y = (sh // 2) - (h // 2)
    root.geometry(f"{w}x{h}+{x}+{y}")
    
    canvas = tk.Canvas(root, width=w, height=h, bg="white", highlightthickness=0)
    dot = canvas.create_oval(15, 15, 35, 35, fill="red", outline="")
    canvas.pack()
    
    root.mainloop()

Thread(target=create_static_overlay, daemon=True).start()

#SETTINGS
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0
screen_w, screen_h = pyautogui.size()

# --- SENSITIVITY ---
SENSITIVITY_X = 1.4
SENSITIVITY_Y = 1.5

# --- THRESHOLDS ---
WINK_THRESHOLD = 0.005       
FREEZE_THRESHOLD = 0.01      
SCROLL_THRESHOLD = 0.03      

# --- SPEEDS ---
SMOOTH_FACTOR = 6            
SNIPER_SMOOTH = 70           
SCROLL_SPEED = 20

#STATE VARIABLES
cal_data = {
    "center": (0,0),
    "min_x": -0.1, "max_x": 0.1,
    "min_y": -0.1, "max_y": 0.1,
    "box_w": 0.1, "box_h": 0.1
}

cal_stage = 0
cal_points_raw = [] 

neutral_nod_dist = 0
neutral_mouth_width = 0
max_mouth_width = 0
dynamic_smile_trigger = 0.03

history_x = deque(maxlen=SNIPER_SMOOTH)
history_y = deque(maxlen=SNIPER_SMOOTH)
last_click_time = 0

#MEDIAPIPE INIT
face_landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(
    mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_faces=1,
    )
)
cap = cv2.VideoCapture(0)

print("-------------------------------------------------")
print("  🔴 STATIC ANCHOR MODE + SCROLL LOCK")
print("  ---------------------------------------------")
print("  1. Look at RED DOT -> Press 'C' (Center)")
print("  2. Look at Corners -> Press 'C'")
print("  ---------------------------------------------")
print("  👀 DRIFT FIX: Look at RED DOT -> Press SPACE")
print("  ---------------------------------------------")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    now = time.time()

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = face_landmarker.detect_for_video(mp_img, int(now * 1000))

    if result.face_landmarks:
        face = result.face_landmarks[0]

        # --- 1. METRICS ---
        eye_l_outer = face[33]; eye_l_inner = face[133]
        iris_l = face[468]

        eye_width = abs(eye_l_outer.x - eye_l_inner.x)
        eye_center_x = (eye_l_outer.x + eye_l_inner.x) / 2
        eye_center_y = (eye_l_outer.y + eye_l_inner.y) / 2

        norm_x = (iris_l.x - eye_center_x) / eye_width
        norm_y = (iris_l.y - eye_center_y) / eye_width
        
        nose = face[1]
        mouth_width = abs(face[61].x - face[291].x)
        right_eye_open = abs(face[386].y - face[374].y)
        left_eye_open = abs(face[159].y - face[145].y)
        current_nod_dist = nose.y - eye_center_y

        cv2.circle(frame, (int(iris_l.x * w), int(iris_l.y * h)), 4, (0, 255, 255), -1)

        # CALIBRATION
        if cal_stage <= 5:
            instructions = [
                "1. LOOK AT RED DOT (Poker Face)",
                "2. SMILE WIDE (Show Teeth)",
                "3. LOOK TOP-LEFT",
                "4. LOOK TOP-RIGHT",
                "5. LOOK BOTTOM-LEFT",
                "6. LOOK BOTTOM-RIGHT"
            ]
            if cal_stage < len(instructions):
                cv2.putText(frame, instructions[cal_stage], (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(frame, "PRESS 'C' TO CAPTURE", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

        # ACTIVE MODE
        elif cal_stage == 6:
            is_winking = (right_eye_open < WINK_THRESHOLD) and (left_eye_open > WINK_THRESHOLD)
            is_frozen = (right_eye_open < FREEZE_THRESHOLD) or (left_eye_open < FREEZE_THRESHOLD)
            is_sniper = mouth_width > dynamic_smile_trigger
            is_scrolling = False

            # --- CLICK ---
            if is_winking and (now - last_click_time) > 0.8:
                pyautogui.click()
                flash_click_signal() 
                last_click_time = now
                cv2.putText(frame, "** CLICK **", (w//2-100, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)

            # --- SCROLL ---
            if not is_frozen: 
                head_tilt = current_nod_dist - neutral_nod_dist
                if head_tilt < -SCROLL_THRESHOLD:
                    pyautogui.scroll(SCROLL_SPEED)
                    cv2.putText(frame, "SCROLL UP", (20, h-80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                    is_scrolling = True
                elif head_tilt > SCROLL_THRESHOLD:
                    pyautogui.scroll(-SCROLL_SPEED)
                    cv2.putText(frame, "SCROLL DOWN", (20, h-80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                    is_scrolling = True

            # --- VISUALS ---
            if is_sniper:
                cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), 10)
                cv2.putText(frame, "🎯 SNIPER", (w//2-100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            elif is_frozen:
                cv2.rectangle(frame, (0, 0), (w, h), (0, 255, 255), 10)
                cv2.putText(frame, "❄️ FROZEN", (w//2-80, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
            elif is_scrolling:
                cv2.putText(frame, "📜 SCROLLING (Cursor Locked)", (w//2-150, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # --- MOVE CURSOR ---
            # Freeze if Winking OR Scrolling
            if not is_frozen and not is_scrolling:
                # 1. Relative to Center
                rel_x = norm_x - cal_data["center"][0]
                rel_y = norm_y - cal_data["center"][1]

                # 2. Normalize by Box Size
                raw_x = rel_x / cal_data["box_w"]
                raw_y = rel_y / cal_data["box_h"]

                # 3. Sensitivity
                final_x = (raw_x * SENSITIVITY_X) + 0.5
                final_y = (raw_y * SENSITIVITY_Y) + 0.5
                
                target_x = int(max(0.0, min(1.0, final_x)) * screen_w)
                target_y = int(max(0.0, min(1.0, final_y)) * screen_h)

                history_x.append(target_x)
                history_y.append(target_y)
                
                smooth_win = SNIPER_SMOOTH if is_sniper else SMOOTH_FACTOR
                smooth_x = int(sum(list(history_x)[-smooth_win:]) / min(len(history_x), smooth_win))
                smooth_y = int(sum(list(history_y)[-smooth_win:]) / min(len(history_y), smooth_win))

                pyautogui.moveTo(smooth_x, smooth_y)

    # INPUTS
    cv2.imshow("Eye Control", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == 32 and cal_stage == 6: # SPACEBAR
        cal_data["center"] = (norm_x, norm_y)
        neutral_nod_dist = current_nod_dist
        print(">>> RE-CENTERED TO RED DOT.")
        cv2.putText(frame, "RE-CENTERED!", (w//2-100, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 3)

    elif key == ord('c'):
        if cal_stage == 0:
            cal_data["center"] = (norm_x, norm_y)
            neutral_nod_dist = current_nod_dist
            neutral_mouth_width = mouth_width
            print("Captured Neutral.")
            cal_stage += 1
        elif cal_stage == 1:
            max_mouth_width = mouth_width
            dynamic_smile_trigger = neutral_mouth_width + (max_mouth_width - neutral_mouth_width) * 0.4
            print(f"Captured Smile. Trig: {dynamic_smile_trigger}")
            cal_stage += 1
        elif cal_stage == 2: cal_points_raw.append((norm_x, norm_y)); print("TL"); cal_stage += 1
        elif cal_stage == 3: cal_points_raw.append((norm_x, norm_y)); print("TR"); cal_stage += 1
        elif cal_stage == 4: cal_points_raw.append((norm_x, norm_y)); print("BL"); cal_stage += 1
        elif cal_stage == 5:
            cal_points_raw.append((norm_x, norm_y))
            print("BR. Processing...")
            
            w1 = abs(cal_points_raw[1][0] - cal_points_raw[0][0])
            w2 = abs(cal_points_raw[3][0] - cal_points_raw[2][0])
            cal_data["box_w"] = (w1 + w2) / 2
            
            h1 = abs(cal_points_raw[2][1] - cal_points_raw[0][1])
            h2 = abs(cal_points_raw[3][1] - cal_points_raw[1][1])
            cal_data["box_h"] = (h1 + h2) / 2
            
            if cal_data["box_w"] == 0: cal_data["box_w"] = 0.1
            if cal_data["box_h"] == 0: cal_data["box_h"] = 0.1
            
            print(f"Done! Box Size: W={cal_data['box_w']:.3f}, H={cal_data['box_h']:.3f}")
            cal_stage += 1

    elif key == ord('r'):
        cal_stage = 0
        cal_points_raw = []
        print("RESET.")
    elif key == 27: break

cap.release()
cv2.destroyAllWindows()
try: root.quit()
except: pass