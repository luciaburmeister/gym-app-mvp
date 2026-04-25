"""
live.py
-------
GymCoach MVP — Phase 3: Live Webcam Feedback (with Auto-Detection)

Opens your webcam, automatically detects which exercise you are
performing, then gives real-time form feedback using the matching
trained classifier.

Controls:
    Q        →  Quit
    R        →  Reset (if detection gets stuck)
    D        →  Toggle developer panel (exercise detection confidence bar)
    1        →  Force: Squat
    2        →  Force: Push-Up
    3        →  Force: Lunge
    4        →  Force: Plank
    0        →  Clear forced exercise (go back to auto-detection)

Usage:
    python live.py
"""

import cv2
import math
import os
import pickle
import numpy as np
from collections import deque
from ultralytics import YOLO


# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────

MODELS_DIR           = "models"
MODEL_WEIGHTS        = "yolov8n-pose.pt"
CONFIDENCE_THRESHOLD = 0.5

# How many frames to accumulate before confirming an exercise
DETECTION_WINDOW    = 20

# Minimum fraction of frames that must agree to confirm exercise
# e.g. 0.7 means 14 out of 20 frames must agree
DETECTION_THRESHOLD = 0.7

# How many frames to smooth the form feedback over
FEEDBACK_WINDOW     = 15

# Keypoint indices
LEFT_SHOULDER  = 5;  RIGHT_SHOULDER = 6
LEFT_ELBOW     = 7;  RIGHT_ELBOW    = 8
LEFT_WRIST     = 9;  RIGHT_WRIST    = 10
LEFT_HIP       = 11; RIGHT_HIP      = 12
LEFT_KNEE      = 13; RIGHT_KNEE     = 14
LEFT_ANKLE     = 15; RIGHT_ANKLE    = 16

# Exercise key mapping for force-override (keys 1–4)
FORCE_EXERCISE_KEYS = {
    ord("1"): "squat",
    ord("2"): "pushup",
    ord("3"): "lunge",
    ord("4"): "plank",
}

EXERCISE_DISPLAY = {
    "squat":      "Squat",
    "pushup":     "Push-Up",
    "lunge":      "Lunge",
    "plank":      "Plank",
    "transition": "Idle",
}

FEEDBACK_MESSAGES = {
    "good_form":          ("Good form!",                              (80, 200, 80)),

    # Squat
    "knees_caving":       ("Knees caving in — push them out",         (60, 120, 255)),
    "chest_down":         ("Chest dropping — keep it up",             (60, 120, 255)),
    "half_rep":           ("Not deep enough — squat lower",           (60, 120, 255)),
    "heels_up":           ("Heels rising — keep feet flat",           (60, 120, 255)),

    # Plank
    "hips_sagging":       ("Hips sagging — engage your core",         (60, 120, 255)),
    "hips_high":          ("Hips too high — lower them down",         (60, 120, 255)),
    "head_position":      ("Head out of alignment — look down",       (60, 120, 255)),
    "head_up":            ("Head tilting up — keep it neutral",       (60, 120, 255)),
    "shoulders_forward":  ("Shoulders too far forward — push back",   (60, 120, 255)),

    # Lunge
    "knee_past_toes":     ("Front knee past toes — step further",     (60, 120, 255)),
    "torso_leaning":      ("Torso leaning — stand upright",           (60, 120, 255)),
    "not_90_degrees":     ("Bend deeper — aim for 90°",               (60, 120, 255)),
    "uneven_hips":        ("Uneven hips — keep them level",           (60, 120, 255)),

    # Push-up
    "elbows_flaring":     ("Elbows flaring out — tuck them in",       (60, 120, 255)),
    "head_dropping":      ("Head dropping — keep it neutral",         (60, 120, 255)),
    "hips_raised":        ("Hips too high — lower your body",         (60, 120, 255)),
}


# ─────────────────────────────────────────────
#  GEOMETRY HELPERS
# ─────────────────────────────────────────────

def get_angle(a, b, c):
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])
    dot = ba[0] * bc[0] + ba[1] * bc[1]
    mag = (ba[0]**2 + ba[1]**2)**0.5 * (bc[0]**2 + bc[1]**2)**0.5
    if mag == 0:
        return 0.0
    return round(math.degrees(math.acos(max(-1.0, min(1.0, dot / mag)))), 2)


def calculate_angles(kp):
    return {
        "left_knee_angle":      get_angle(kp[LEFT_HIP],       kp[LEFT_KNEE],      kp[LEFT_ANKLE]),
        "right_knee_angle":     get_angle(kp[RIGHT_HIP],      kp[RIGHT_KNEE],     kp[RIGHT_ANKLE]),
        "left_hip_angle":       get_angle(kp[LEFT_SHOULDER],  kp[LEFT_HIP],       kp[LEFT_KNEE]),
        "right_hip_angle":      get_angle(kp[RIGHT_SHOULDER], kp[RIGHT_HIP],      kp[RIGHT_KNEE]),
        "left_elbow_angle":     get_angle(kp[LEFT_SHOULDER],  kp[LEFT_ELBOW],     kp[LEFT_WRIST]),
        "right_elbow_angle":    get_angle(kp[RIGHT_SHOULDER], kp[RIGHT_ELBOW],    kp[RIGHT_WRIST]),
        "left_shoulder_angle":  get_angle(kp[LEFT_HIP],       kp[LEFT_SHOULDER],  kp[LEFT_ELBOW]),
        "right_shoulder_angle": get_angle(kp[RIGHT_HIP],      kp[RIGHT_SHOULDER], kp[RIGHT_ELBOW]),
    }


def extract_features(kp):
    angles = calculate_angles(kp)
    return (
        [angles["left_knee_angle"],
         angles["right_knee_angle"],
         angles["left_hip_angle"],
         angles["right_hip_angle"],
         angles["left_elbow_angle"],
         angles["right_elbow_angle"],
         angles["left_shoulder_angle"],
         angles["right_shoulder_angle"]] +
        [float(kp[i][0]) for i in range(17)] +
        [float(kp[i][1]) for i in range(17)]
    )


# ─────────────────────────────────────────────
#  MODEL LOADING
# ─────────────────────────────────────────────

def load_models():
    models = {}

    det_path = os.path.join(MODELS_DIR, "exercise_detector.pkl")
    if not os.path.exists(det_path):
        raise FileNotFoundError(
            "exercise_detector.pkl not found. "
            "Run train_exercise_detector.py first."
        )
    with open(det_path, "rb") as f:
        models["detector"] = pickle.load(f)
    print("  ✓  Loaded exercise detector")

    for exercise in ["squat", "pushup", "lunge", "plank"]:
        path = os.path.join(MODELS_DIR, f"{exercise}_model.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                models[exercise] = pickle.load(f)
            print(f"  ✓  Loaded {exercise} form model")
        else:
            print(f"  [!] Missing: {path}")

    return models


# ─────────────────────────────────────────────
#  DRAWING HELPERS
# ─────────────────────────────────────────────

def draw_confidence_bar(frame, x, y, width, height, value, color, bg_color=(60, 60, 60)):
    """Draw a filled confidence bar from (x,y) with the given dimensions."""
    cv2.rectangle(frame, (x, y), (x + width, y + height), bg_color, -1)
    filled = int(width * max(0.0, min(1.0, value)))
    if filled > 0:
        cv2.rectangle(frame, (x, y), (x + filled, y + height), color, -1)


def draw_ui(frame, state):
    h, w = frame.shape[:2]

    # ── Top bar background ──
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 75), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    # ── Bottom bar background ──
    # Taller when feedback confidence bar is shown
    bottom_bar_h = 100 if (state["confirmed_exercise"] and not state["no_person"]) else 80
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - bottom_bar_h), (w, h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    no_person          = state["no_person"]
    exercise           = state["confirmed_exercise"]
    detecting          = state["detecting"]
    detection_pct      = state["detection_pct"]
    feedback_text      = state["feedback_text"]
    feedback_color     = state["feedback_color"]
    form_confidence    = state["form_confidence"]
    feedback_confidence = state["feedback_confidence"]
    show_dev_panel     = state["show_dev_panel"]
    forced_exercise    = state["forced_exercise"]

    # ── TOP BAR ──

    if no_person:
        cv2.putText(frame, "Step into frame to begin",
                    (20, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (140, 140, 140), 2)

    elif detecting:
        label = f"Detecting exercise...  {int(detection_pct * 100)}%"
        cv2.putText(frame, label,
                    (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 180, 60), 2)

        # Detection progress bar (only shown while detecting — this is not the
        # developer confidence bar, it's the "scanning" indicator every user sees)
        bx, by, bw, bh = 20, 58, 300, 8
        draw_confidence_bar(frame, bx, by, bw, bh, detection_pct, (200, 180, 60))

    else:
        ex_label = EXERCISE_DISPLAY.get(exercise, exercise)

        # If exercise was forced, show a small "MANUAL" badge next to the name
        if forced_exercise:
            cv2.putText(frame, ex_label,
                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 200, 60), 2)
            badge = "MANUAL"
            ts = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            bx = 20 + cv2.getTextSize(ex_label, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)[0][0] + 10
            cv2.putText(frame, badge, (bx, 46),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 60), 1)
        else:
            cv2.putText(frame, ex_label, 
                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

        # ── DEVELOPER PANEL (toggle with D) ──
        # Shows the exercise detector's confidence bar — hidden by default
        if show_dev_panel and form_confidence > 0:
            dev_label = f"DEV  detector conf: {int(form_confidence * 100)}%"
            cv2.putText(frame, dev_label,
                        (w - 280, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 160, 0), 1)
            draw_confidence_bar(frame, w - 280, 38, 260, 7,
                                form_confidence, (255, 160, 0))

    # ── BOTTOM BAR: feedback text + feedback confidence bar ──

    if feedback_text and not no_person and not detecting:
        # Feedback text
        cv2.putText(frame, feedback_text,
                    (20, h - bottom_bar_h + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, feedback_color, 2)

        # Feedback confidence bar + label
        if feedback_confidence > 0:
            pct_label = f"confidence  {int(feedback_confidence * 100)}%"
            cv2.putText(frame, pct_label,
                        (20, h - bottom_bar_h + 52),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (160, 160, 160), 1)
            draw_confidence_bar(frame, 20, h - bottom_bar_h + 60,
                                220, 7, feedback_confidence, feedback_color)

    # ── Exercise force buttons hint (always visible at bottom right) ──
    hint_lines = [
        "1=Squat  2=Push-Up  3=Lunge  4=Plank  0=Auto",
        "D=Dev panel   R=Reset   Q=Quit",
    ]
    for i, line in enumerate(hint_lines):
        ts = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)[0]
        cv2.putText(frame, line,
                    (w - ts[0] - 10, h - 22 + i * 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (100, 100, 100), 1)

    return frame


# ─────────────────────────────────────────────
#  MAIN LOOP
# ─────────────────────────────────────────────

def run():
    print("\n Loading models...")
    models = load_models()

    print("\n Loading YOLOv8-pose...")
    yolo = YOLO(MODEL_WEIGHTS)
    print(" Ready!\n")
    print(" Controls:")
    print("   1/2/3/4  →  Force exercise (Squat / Push-Up / Lunge / Plank)")
    print("   0        →  Back to auto-detection")
    print("   D        →  Toggle developer confidence panel")
    print("   R        →  Reset")
    print("   Q        →  Quit\n")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[!] Could not open webcam.")
        return

    detection_buffer   = deque(maxlen=DETECTION_WINDOW)
    feedback_buffer    = deque(maxlen=FEEDBACK_WINDOW)
    confirmed_exercise  = None
    forced_exercise     = None   # set when user presses 1–4
    feedback_text       = ""
    feedback_color      = (200, 200, 200)
    form_confidence     = 0.0
    feedback_confidence = 0.0   # confidence of the current feedback label
    show_dev_panel      = False  # toggled with D
    detector            = models["detector"]

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        key = cv2.waitKey(1) & 0xFF

        # ── Quit ──
        if key == ord("q") or key == ord("Q"):
            break

        # ── Reset ──
        if key == ord("r") or key == ord("R"):
            detection_buffer.clear()
            feedback_buffer.clear()
            confirmed_exercise  = None
            forced_exercise     = None
            feedback_text       = ""
            form_confidence     = 0.0
            feedback_confidence = 0.0
            print(" Reset — back to auto-detection.")

        # ── Toggle developer panel (exercise detection confidence) ──
        if key == ord("d") or key == ord("D"):
            show_dev_panel = not show_dev_panel
            print(f" Developer panel {'ON' if show_dev_panel else 'OFF'}.")

        # ── Force exercise override (keys 1–4) ──
        if key in FORCE_EXERCISE_KEYS:
            forced_exercise    = FORCE_EXERCISE_KEYS[key]
            confirmed_exercise = forced_exercise
            detection_buffer.clear()
            feedback_buffer.clear()
            feedback_text       = ""
            feedback_confidence = 0.0
            print(f" Forced exercise: {EXERCISE_DISPLAY.get(forced_exercise, forced_exercise)}")

        # ── Clear force override (key 0) ──
        if key == ord("0"):
            forced_exercise    = None
            confirmed_exercise = None
            detection_buffer.clear()
            feedback_buffer.clear()
            feedback_text       = ""
            form_confidence     = 0.0
            feedback_confidence = 0.0
            print(" Cleared forced exercise — back to auto-detection.")

        # ── YOLO inference ──
        results   = yolo(frame, verbose=False)
        no_person = True

        if results and results[0].keypoints is not None:
            if len(results[0].keypoints.xy) > 0:
                kp   = results[0].keypoints.xy[0].cpu().numpy()
                conf = results[0].keypoints.conf[0].cpu().numpy()

                hip_visible = (
                    conf[LEFT_HIP]  >= CONFIDENCE_THRESHOLD and
                    conf[RIGHT_HIP] >= CONFIDENCE_THRESHOLD
                )

                if hip_visible:
                    no_person = False
                    frame     = results[0].plot(img=frame)
                    features  = extract_features(kp)

                    # ── Step 1: Detect the exercise ──
                    # Skip auto-detection if the user has forced an exercise
                    if confirmed_exercise is None and forced_exercise is None:
                        det_proba = detector["model"].predict_proba([features])[0]
                        det_label = detector["encoder"].inverse_transform(
                            [det_proba.argmax()]
                        )[0]
                        detection_buffer.append(det_label)

                        # Store detection confidence for dev panel
                        form_confidence = det_proba.max()

                        if len(detection_buffer) == DETECTION_WINDOW:
                            counts   = {ex: detection_buffer.count(ex)
                                        for ex in set(detection_buffer)}
                            top_ex   = max(counts, key=counts.get)
                            top_frac = counts[top_ex] / DETECTION_WINDOW

                            if top_frac >= DETECTION_THRESHOLD and top_ex != "transition":
                                confirmed_exercise = top_ex
                                feedback_buffer.clear()
                                print(f" Detected: {EXERCISE_DISPLAY.get(top_ex, top_ex)}")
                            else:
                                # Not confident yet — keep collecting
                                detection_buffer.clear()

                    # ── Step 2: Give form feedback ──
                    elif confirmed_exercise in models:
                        form_data = models[confirmed_exercise]
                        proba     = form_data["model"].predict_proba([features])[0]
                        pred_idx  = proba.argmax()

                        # form_confidence is now used only for the dev panel
                        form_confidence     = proba[pred_idx]
                        # feedback_confidence is the confidence shown to users
                        # in the bottom bar — we store the raw per-frame value
                        # and show it smoothed via the buffer
                        feedback_confidence = proba[pred_idx]

                        feedback_buffer.append(pred_idx)
                        smoothed = max(set(feedback_buffer),
                                       key=feedback_buffer.count)
                        label    = form_data["encoder"].inverse_transform([smoothed])[0]

                        if label in FEEDBACK_MESSAGES:
                            feedback_text, feedback_color = FEEDBACK_MESSAGES[label]
                        else:
                            feedback_text  = label.replace("_", " ").title()
                            feedback_color = (60, 120, 255)

        # ── Reset if person leaves frame (but keep forced exercise in memory) ──
        if no_person and confirmed_exercise is not None:
            confirmed_exercise  = None
            # If exercise was forced, we'll restore it when the person comes back
            # only reset the auto-detected one
            if not forced_exercise:
                detection_buffer.clear()
            feedback_buffer.clear()
            feedback_text       = ""
            form_confidence     = 0.0
            feedback_confidence = 0.0

        # If person comes back and exercise was forced, restore it immediately
        if not no_person and forced_exercise and confirmed_exercise is None:
            confirmed_exercise = forced_exercise

        detecting     = (confirmed_exercise is None and
                         not no_person and
                         len(detection_buffer) > 0)
        detection_pct = len(detection_buffer) / DETECTION_WINDOW if detecting else 0.0

        state = {
            "no_person":           no_person,
            "detecting":           detecting,
            "detection_pct":       detection_pct,
            "confirmed_exercise":  confirmed_exercise,
            "feedback_text":       feedback_text,
            "feedback_color":      feedback_color,
            "form_confidence":     form_confidence,
            "feedback_confidence": feedback_confidence,
            "show_dev_panel":      show_dev_panel,
            "forced_exercise":     forced_exercise,
        }

        frame = draw_ui(frame, state)
        cv2.imshow("GymCoach — Live Feedback", frame)

    cap.release()
    cv2.destroyAllWindows()
    print("\n Session ended.\n")


if __name__ == "__main__":
    run()