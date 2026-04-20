"""
live.py
-------
GymCoach MVP — Phase 3: Live Webcam Feedback (with Auto-Detection)

Opens your webcam, automatically detects which exercise you are
performing, then gives real-time form feedback using the matching
trained classifier.

Controls:
    Q  →  Quit
    R  →  Reset (if detection gets stuck)

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
#  DRAWING
# ─────────────────────────────────────────────

def draw_ui(frame, state):
    h, w = frame.shape[:2]

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 75), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 80), (w, h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    no_person       = state["no_person"]
    exercise        = state["confirmed_exercise"]
    detecting       = state["detecting"]
    detection_pct   = state["detection_pct"]
    feedback_text   = state["feedback_text"]
    feedback_color  = state["feedback_color"]
    form_confidence = state["form_confidence"]

    if no_person:
        cv2.putText(frame, "Step into frame to begin",
                    (20, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (140, 140, 140), 2)

    elif detecting:
        label = f"Detecting exercise...  {int(detection_pct * 100)}%"
        cv2.putText(frame, label,
                    (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 180, 60), 2)

        bx, by, bw, bh = 20, 58, 300, 8
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (60, 60, 60), -1)
        cv2.rectangle(frame, (bx, by),
                      (bx + int(bw * detection_pct), by + bh),
                      (200, 180, 60), -1)

    else:
        ex_label = EXERCISE_DISPLAY.get(exercise, exercise)
        cv2.putText(frame, ex_label,
                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

        if form_confidence > 0:
            conf_text = f"{int(form_confidence * 100)}%"
            ts = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)[0]
            cv2.putText(frame, conf_text,
                        (w - ts[0] - 20, 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 180, 180), 2)

    if feedback_text and not no_person and not detecting:
        cv2.putText(frame, feedback_text,
                    (20, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    feedback_color, 2)

    hint = "R=Reset  Q=Quit"
    ts   = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
    cv2.putText(frame, hint,
                (w - ts[0] - 10, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1)

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
    print(" Opening webcam — stand in frame to begin. Q=Quit  R=Reset\n")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[!] Could not open webcam.")
        return

    detection_buffer   = deque(maxlen=DETECTION_WINDOW)
    feedback_buffer    = deque(maxlen=FEEDBACK_WINDOW)
    confirmed_exercise = None
    feedback_text      = ""
    feedback_color     = (200, 200, 200)
    form_confidence    = 0.0
    detector           = models["detector"]

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == ord("Q"):
            break
        if key == ord("r") or key == ord("R"):
            detection_buffer.clear()
            feedback_buffer.clear()
            confirmed_exercise = None
            feedback_text      = ""
            form_confidence    = 0.0
            print(" Reset.")

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
                    if confirmed_exercise is None:
                        det_proba = detector["model"].predict_proba([features])[0]
                        det_label = detector["encoder"].inverse_transform(
                            [det_proba.argmax()]
                        )[0]
                        detection_buffer.append(det_label)

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
                        form_confidence = proba[pred_idx]

                        feedback_buffer.append(pred_idx)
                        smoothed = max(set(feedback_buffer),
                                       key=feedback_buffer.count)
                        label    = form_data["encoder"].inverse_transform([smoothed])[0]

                        if label in FEEDBACK_MESSAGES:
                            feedback_text, feedback_color = FEEDBACK_MESSAGES[label]
                        else:
                            feedback_text  = label.replace("_", " ").title()
                            feedback_color = (60, 120, 255)

        # Reset if person leaves the frame
        if no_person and confirmed_exercise is not None:
            confirmed_exercise = None
            detection_buffer.clear()
            feedback_buffer.clear()
            feedback_text   = ""
            form_confidence = 0.0

        detecting     = (confirmed_exercise is None and
                         not no_person and
                         len(detection_buffer) > 0)
        detection_pct = len(detection_buffer) / DETECTION_WINDOW if detecting else 0.0

        state = {
            "no_person":          no_person,
            "detecting":          detecting,
            "detection_pct":      detection_pct,
            "confirmed_exercise": confirmed_exercise,
            "feedback_text":      feedback_text,
            "feedback_color":     feedback_color,
            "form_confidence":    form_confidence,
        }

        frame = draw_ui(frame, state)
        cv2.imshow("GymCoach — Live Feedback", frame)

    cap.release()
    cv2.destroyAllWindows()
    print("\n Session ended.\n")


if __name__ == "__main__":
    run()
