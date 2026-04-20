"""
app.py
------
GymCoach MVP — Web Interface Backend

Flask server that:
  1. Runs the full AI pipeline in a background thread
     (YOLOv8-pose → exercise detection → form classification)
  2. Streams the annotated video feed to the browser via MJPEG
  3. Exposes a /status endpoint the browser polls for feedback data

Usage:
    python app.py
Then open:  http://localhost:5000
"""

import cv2
import math
import os
import pickle
import threading
import time
from collections import deque

from flask import Flask, Response, jsonify, render_template
from ultralytics import YOLO


# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────

MODELS_DIR           = "models"
MODEL_WEIGHTS        = "yolov8n-pose.pt"
CONFIDENCE_THRESHOLD = 0.5
DETECTION_WINDOW     = 20
DETECTION_THRESHOLD  = 0.7
FEEDBACK_WINDOW      = 15

# Keypoint indices
LEFT_SHOULDER  = 5;  RIGHT_SHOULDER = 6
LEFT_ELBOW     = 7;  RIGHT_ELBOW    = 8
LEFT_WRIST     = 9;  RIGHT_WRIST    = 10
LEFT_HIP       = 11; RIGHT_HIP      = 12
LEFT_KNEE      = 13; RIGHT_KNEE     = 14
LEFT_ANKLE     = 15; RIGHT_ANKLE    = 16

EXERCISE_DISPLAY = {
    "squat":  "Squat",
    "pushup": "Push-Up",
    "lunge":  "Lunge",
    "plank":  "Plank",
}

FEEDBACK_MESSAGES = {
    "good_form":         "Good form — keep it up!",
    "knees_caving":      "Knees caving in — push them out",
    "chest_down":        "Chest dropping — keep it up",
    "half_rep":          "Not deep enough — squat lower",
    "heels_up":          "Heels rising — keep feet flat",
    "hips_sagging":      "Hips sagging — engage your core",
    "hips_high":         "Hips too high — lower them down",
    "head_position":     "Head out of alignment — look down",
    "head_up":           "Head tilting up — keep it neutral",
    "shoulders_forward": "Shoulders too far forward — push back",
    "knee_past_toes":    "Front knee past toes — step further",
    "torso_leaning":     "Torso leaning — stand upright",
    "not_90_degrees":    "Bend deeper — aim for 90°",
    "uneven_hips":       "Uneven hips — keep them level",
    "elbows_flaring":    "Elbows flaring out — tuck them in",
    "head_dropping":     "Head dropping — keep it neutral",
    "hips_raised":       "Hips too high — lower your body",
}


# ─────────────────────────────────────────────
#  GLOBAL STATE (shared between AI thread + Flask)
# ─────────────────────────────────────────────

state_lock = threading.Lock()

state = {
    "status":           "waiting",   # waiting | detecting | active
    "exercise":         None,
    "feedback_key":     None,
    "feedback_message": "",
    "form_confidence":  0.0,
    "detection_pct":    0.0,
    "is_good_form":     False,
}

# The latest JPEG frame to stream to the browser
frame_lock    = threading.Lock()
latest_frame  = None


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
        [angles["left_knee_angle"],  angles["right_knee_angle"],
         angles["left_hip_angle"],   angles["right_hip_angle"],
         angles["left_elbow_angle"], angles["right_elbow_angle"],
         angles["left_shoulder_angle"], angles["right_shoulder_angle"]] +
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
            "exercise_detector.pkl not found — run train_exercise_detector.py first."
        )
    with open(det_path, "rb") as f:
        models["detector"] = pickle.load(f)
    print("  ✓  Exercise detector loaded")

    for ex in ["squat", "pushup", "lunge", "plank"]:
        path = os.path.join(MODELS_DIR, f"{ex}_model.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                models[ex] = pickle.load(f)
            print(f"  ✓  {ex} form model loaded")

    return models


# ─────────────────────────────────────────────
#  AI PROCESSING THREAD
# ─────────────────────────────────────────────

def ai_loop(models):
    global latest_frame, state

    yolo = YOLO(MODEL_WEIGHTS)
    print("  ✓  YOLOv8-pose ready")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[!] Could not open webcam.")
        return

    detection_buffer   = deque(maxlen=DETECTION_WINDOW)
    feedback_buffer    = deque(maxlen=FEEDBACK_WINDOW)
    confirmed_exercise = None
    detector           = models["detector"]

    print("\n  Webcam open — visit http://localhost:5000\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        frame    = cv2.flip(frame, 1)
        results  = yolo(frame, verbose=False)
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

                    # Draw skeleton on frame
                    frame    = results[0].plot(img=frame)
                    features = extract_features(kp)

                    # ── Exercise detection ──
                    if confirmed_exercise is None:
                        det_proba = detector["model"].predict_proba([features])[0]
                        det_label = detector["encoder"].inverse_transform(
                            [det_proba.argmax()]
                        )[0]
                        detection_buffer.append(det_label)

                        det_pct = len(detection_buffer) / DETECTION_WINDOW

                        if len(detection_buffer) == DETECTION_WINDOW:
                            counts   = {ex: detection_buffer.count(ex)
                                        for ex in set(detection_buffer)}
                            top_ex   = max(counts, key=counts.get)
                            top_frac = counts[top_ex] / DETECTION_WINDOW

                            if top_frac >= DETECTION_THRESHOLD and top_ex != "transition":
                                confirmed_exercise = top_ex
                                feedback_buffer.clear()
                                print(f"  → Detected: {top_ex}")
                            else:
                                detection_buffer.clear()

                        with state_lock:
                            state["status"]        = "detecting"
                            state["detection_pct"] = det_pct
                            state["exercise"]      = None

                    # ── Form feedback ──
                    elif confirmed_exercise in models:
                        form_data = models[confirmed_exercise]
                        proba     = form_data["model"].predict_proba([features])[0]
                        pred_idx  = proba.argmax()
                        confidence = float(proba[pred_idx])

                        feedback_buffer.append(pred_idx)
                        smoothed = max(set(feedback_buffer), key=feedback_buffer.count)
                        label    = form_data["encoder"].inverse_transform([smoothed])[0]
                        message  = FEEDBACK_MESSAGES.get(
                            label, label.replace("_", " ").title()
                        )

                        with state_lock:
                            state["status"]           = "active"
                            state["exercise"]         = confirmed_exercise
                            state["feedback_key"]     = label
                            state["feedback_message"] = message
                            state["form_confidence"]  = confidence
                            state["is_good_form"]     = (label == "good_form")

        # Person left frame — reset
        if no_person:
            if confirmed_exercise is not None:
                confirmed_exercise = None
                detection_buffer.clear()
                feedback_buffer.clear()
            with state_lock:
                state["status"]          = "waiting"
                state["exercise"]        = None
                state["feedback_message"] = ""
                state["form_confidence"] = 0.0
                state["detection_pct"]   = 0.0

        # Encode frame as JPEG and store for streaming
        _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        with frame_lock:
            latest_frame = jpeg.tobytes()


# ─────────────────────────────────────────────
#  FLASK APP
# ─────────────────────────────────────────────

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    """MJPEG stream of the annotated webcam feed."""
    def generate():
        while True:
            with frame_lock:
                frame = latest_frame
            if frame:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" +
                    frame +
                    b"\r\n"
                )
            time.sleep(0.03)  # ~30fps

    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/status")
def status():
    """JSON snapshot of the current AI state — polled by the browser."""
    with state_lock:
        return jsonify(dict(state))


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("\n Loading models...")
    models = load_models()

    # Start AI thread
    ai_thread = threading.Thread(target=ai_loop, args=(models,), daemon=True)
    ai_thread.start()

    # Give the webcam a moment to open
    time.sleep(1)

    print("\n Starting web server...")
    print(" Open your browser at:  http://localhost:5000\n")

    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
