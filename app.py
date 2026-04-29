"""
app.py
------
GymCoach MVP — this is the web interface backend 

Flask server that:
  1. Runs the full AI pipeline in a background thread
     (YOLOv8-pose → exercise detection → form classification)
  2. Streams the annotated video feed to the browser via MJPEG
  3. Exposes a /status endpoint the browser polls for feedback data
  4. Exposes a /force_exercise endpoint so the UI can override detection

v3 changes:
  - Added RepCounter class for squat, pushup, lunge (angle state machine)
  - Added PlankTimer class for plank hold time tracking
  - /status now returns rep_count and hold_time
  - Reset rep counters when exercise changes

Usage:
    python app.py
Then open:  http://localhost:8080
"""

import cv2
import math
import os
import pickle
import threading
import time
import warnings
from collections import deque

warnings.filterwarnings("ignore", category=UserWarning)

from flask import Flask, Response, jsonify, render_template, request
from ultralytics import YOLO


#configuration constants

MODELS_DIR           = "models"
MODEL_WEIGHTS        = "yolov8n-pose.pt"
CONFIDENCE_THRESHOLD = 0.5
DETECTION_WINDOW     = 20
DETECTION_THRESHOLD  = 0.7
FEEDBACK_WINDOW      = 15
FORM_CONFIDENCE_THRESHOLD = 0.55

#keypoint indices
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


#rep counter

class RepCounter:
    """
    Counts reps for squat, pushup, and lunge using a simple angle state machine.

    How it works:
      - We track one key joint angle per exercise.
      - When the angle drops below DOWN_THRESHOLD  → person is at the bottom of the rep.
      - When the angle rises above UP_THRESHOLD    → person is back at the top = 1 rep counted.
      - We require the angle to fully reach the bottom before we count the return.
      - A small deadband between the two thresholds prevents flickering.

    Angle reference (approximate):
      Squat  — knee angle:  ~170° standing,  ~90° at bottom
      Pushup — elbow angle: ~160° extended, ~90° at bottom
      Lunge  — front knee:  ~170° standing,  ~90° at bottom
    """

    # (down_threshold, up_threshold) per exercise
    THRESHOLDS = {
        "squat":  (110, 150),   # knee angle
        "pushup": (100, 140),   # elbow angle
        "lunge":  (110, 150),   # front knee angle
    }

    def __init__(self, exercise):
        self.exercise      = exercise
        self.count         = 0
        self.state         = "up"      # "up" or "down"
        thresholds         = self.THRESHOLDS.get(exercise, (110, 150))
        self.down_thresh   = thresholds[0]
        self.up_thresh     = thresholds[1]

    def update(self, kp, conf):
        """
        Pass in the keypoint array and confidence array from YOLO.
        Returns the current rep count.
        """
        angle = self._get_key_angle(kp, conf)
        if angle is None:
            return self.count

        if self.state == "up" and angle < self.down_thresh:
            self.state = "down"

        elif self.state == "down" and angle > self.up_thresh:
            self.state = "up"
            self.count += 1

        return self.count

    def reset(self):
        self.count = 0
        self.state = "up"

    def _get_key_angle(self, kp, conf):
        """Returns the relevant angle for rep counting, or None if keypoints not visible."""
        if self.exercise == "squat":
            # Use the average of both knee angles
            if (conf[LEFT_HIP]   >= CONFIDENCE_THRESHOLD and
                conf[LEFT_KNEE]  >= CONFIDENCE_THRESHOLD and
                conf[LEFT_ANKLE] >= CONFIDENCE_THRESHOLD):
                left  = get_angle(kp[LEFT_HIP],  kp[LEFT_KNEE],  kp[LEFT_ANKLE])
                right = get_angle(kp[RIGHT_HIP], kp[RIGHT_KNEE], kp[RIGHT_ANKLE])
                return (left + right) / 2
            return None

        elif self.exercise == "pushup":
            # Use the elbow angle — take whichever side is more visible
            if (conf[LEFT_SHOULDER] >= CONFIDENCE_THRESHOLD and
                conf[LEFT_ELBOW]    >= CONFIDENCE_THRESHOLD and
                conf[LEFT_WRIST]    >= CONFIDENCE_THRESHOLD):
                return get_angle(kp[LEFT_SHOULDER], kp[LEFT_ELBOW], kp[LEFT_WRIST])
            if (conf[RIGHT_SHOULDER] >= CONFIDENCE_THRESHOLD and
                conf[RIGHT_ELBOW]    >= CONFIDENCE_THRESHOLD and
                conf[RIGHT_WRIST]    >= CONFIDENCE_THRESHOLD):
                return get_angle(kp[RIGHT_SHOULDER], kp[RIGHT_ELBOW], kp[RIGHT_WRIST])
            return None

        elif self.exercise == "lunge":
            # Use the front (lower) knee — whichever knee is more bent
            left_angle  = None
            right_angle = None
            if (conf[LEFT_HIP]   >= CONFIDENCE_THRESHOLD and
                conf[LEFT_KNEE]  >= CONFIDENCE_THRESHOLD and
                conf[LEFT_ANKLE] >= CONFIDENCE_THRESHOLD):
                left_angle = get_angle(kp[LEFT_HIP], kp[LEFT_KNEE], kp[LEFT_ANKLE])
            if (conf[RIGHT_HIP]   >= CONFIDENCE_THRESHOLD and
                conf[RIGHT_KNEE]  >= CONFIDENCE_THRESHOLD and
                conf[RIGHT_ANKLE] >= CONFIDENCE_THRESHOLD):
                right_angle = get_angle(kp[RIGHT_HIP], kp[RIGHT_KNEE], kp[RIGHT_ANKLE])

            # The front (working) knee is the one with the smaller angle
            angles = [a for a in [left_angle, right_angle] if a is not None]
            return min(angles) if angles else None

        return None


#plank timer 

class PlankTimer:
    """
    Tracks how long the user has been holding a plank.
    Starts when the exercise is confirmed as plank.
    Pauses if the person leaves the frame.
    """
    def __init__(self):
        self.start_time  = None
        self.hold_time   = 0.0   # seconds held so far
        self.is_running  = False

    def start(self):
        if not self.is_running:
            self.start_time = time.time()
            self.is_running = True

    def pause(self):
        if self.is_running:
            self.hold_time  += time.time() - self.start_time
            self.is_running  = False
            self.start_time  = None

    def reset(self):
        self.start_time = None
        self.hold_time  = 0.0
        self.is_running = False

    def get(self):
        """Returns total seconds held including currently running segment."""
        if self.is_running and self.start_time is not None:
            return self.hold_time + (time.time() - self.start_time)
        return self.hold_time


#global state

state_lock = threading.Lock()

state = {
    "status":              "waiting",
    "exercise":            None,
    "feedback_key":        None,
    "feedback_message":    "",
    "form_confidence":     0.0,
    "feedback_confidence": 0.0,
    "detection_pct":       0.0,
    "is_good_form":        False,
    "forced_exercise":     None,
    "rep_count":           0,
    "hold_time":           0.0,
}

frame_lock   = threading.Lock()
latest_frame = None


# geometry helpers 


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


def extract_features_detector(kp):
    """52 features for the exercise detector."""
    angles = calculate_angles(kp)

    base = (
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

    asymmetry = [
        abs(angles["left_knee_angle"]     - angles["right_knee_angle"]),
        abs(angles["left_hip_angle"]      - angles["right_hip_angle"]),
        abs(angles["left_shoulder_angle"] - angles["right_shoulder_angle"]),
        abs(angles["left_elbow_angle"]    - angles["right_elbow_angle"]),
        abs(float(kp[LEFT_HIP][0])        - float(kp[RIGHT_HIP][0])),
        abs(float(kp[LEFT_KNEE][0])       - float(kp[RIGHT_KNEE][0])),
        abs(float(kp[LEFT_ANKLE][0])      - float(kp[RIGHT_ANKLE][0])),
        abs(float(kp[LEFT_KNEE][1])       - float(kp[RIGHT_KNEE][1])),
        abs(float(kp[LEFT_ANKLE][1])      - float(kp[RIGHT_ANKLE][1])),
    ]

    shoulder_mid_x = (float(kp[LEFT_SHOULDER][0]) + float(kp[RIGHT_SHOULDER][0])) / 2
    shoulder_mid_y = (float(kp[LEFT_SHOULDER][1]) + float(kp[RIGHT_SHOULDER][1])) / 2
    hip_mid_x      = (float(kp[LEFT_HIP][0])      + float(kp[RIGHT_HIP][0]))      / 2
    hip_mid_y      = (float(kp[LEFT_HIP][1])      + float(kp[RIGHT_HIP][1]))      / 2

    dx = shoulder_mid_x - hip_mid_x
    dy = shoulder_mid_y - hip_mid_y
    torso_angle = math.degrees(math.atan2(abs(dy), abs(dx) + 1e-6))

    return base + asymmetry + [torso_angle]


def extract_features_form(kp):
    """42 features for the per-exercise form classifiers."""
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


#model loading 


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



#ai processing thread 


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

    # Rep counting
    rep_counter  = None
    plank_timer  = PlankTimer()

    print("\n  Webcam open — visit http://localhost:8080\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        frame = cv2.flip(frame, 1)

        with state_lock:
            forced_exercise = state["forced_exercise"]

        # Handle forced exercise change
        if forced_exercise is not None and confirmed_exercise != forced_exercise:
            confirmed_exercise = forced_exercise
            detection_buffer.clear()
            feedback_buffer.clear()
            rep_counter  = _make_counter(forced_exercise, plank_timer)

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
                    feat_det  = extract_features_detector(kp)
                    feat_form = extract_features_form(kp)

                    # exercise detection (only if not already confirmed or forced)
                    if confirmed_exercise is None and forced_exercise is None:
                        det_proba = detector["model"].predict_proba([feat_det])[0]
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
                                rep_counter = _make_counter(top_ex, plank_timer)
                                print(f"  → Detected: {top_ex}")
                            else:
                                detection_buffer.clear()

                        with state_lock:
                            state["status"]        = "detecting"
                            state["detection_pct"] = det_pct
                            state["exercise"]      = None

                    # form feedback and rep counting 
                    elif confirmed_exercise in models:

                        # ── Rep counting ──
                        current_reps      = 0
                        current_hold_time = 0.0

                        if confirmed_exercise == "plank":
                            plank_timer.start()
                            current_hold_time = plank_timer.get()
                        elif rep_counter is not None:
                            current_reps = rep_counter.update(kp, conf)

                        # form feedback
                        form_data  = models[confirmed_exercise]
                        proba      = form_data["model"].predict_proba([feat_form])[0]
                        pred_idx   = proba.argmax()
                        confidence = float(proba[pred_idx])

                        feedback_buffer.append(pred_idx)
                        smoothed = max(set(feedback_buffer), key=feedback_buffer.count)
                        label    = form_data["encoder"].inverse_transform([smoothed])[0]
                        message  = FEEDBACK_MESSAGES.get(
                            label, label.replace("_", " ").title()
                        )

                        if confidence < FORM_CONFIDENCE_THRESHOLD and label != "good_form":
                            label   = "good_form"
                            message = FEEDBACK_MESSAGES["good_form"]

                        with state_lock:
                            state["status"]              = "active"
                            state["exercise"]            = confirmed_exercise
                            state["feedback_key"]        = label
                            state["feedback_message"]    = message
                            state["form_confidence"]     = confidence
                            state["feedback_confidence"] = confidence
                            state["is_good_form"]        = (label == "good_form")
                            state["rep_count"]           = current_reps
                            state["hold_time"]           = round(current_hold_time, 1)

        if no_person:
            # Pause plank timer if person leaves
            if confirmed_exercise == "plank":
                plank_timer.pause()

            if confirmed_exercise is not None and forced_exercise is None:
                confirmed_exercise = None
                detection_buffer.clear()
                feedback_buffer.clear()
                rep_counter = None

            with state_lock:
                state["status"]              = "waiting"
                state["exercise"]            = None
                state["feedback_message"]    = ""
                state["form_confidence"]     = 0.0
                state["feedback_confidence"] = 0.0
                state["detection_pct"]       = 0.0

        _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        with frame_lock:
            latest_frame = jpeg.tobytes()


def _make_counter(exercise, plank_timer):
    """Creates a fresh RepCounter or resets the PlankTimer depending on exercise."""
    plank_timer.reset()
    if exercise == "plank":
        return None
    return RepCounter(exercise)


# flask app


app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
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
            time.sleep(0.03)
    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/status")
def status():
    with state_lock:
        return jsonify(dict(state))


@app.route("/force_exercise", methods=["POST"])
def force_exercise():
    data     = request.get_json(silent=True) or {}
    exercise = data.get("exercise")

    valid = {"squat", "pushup", "lunge", "plank", None}
    if exercise not in valid:
        return jsonify({"ok": False, "error": "unknown exercise"}), 400

    with state_lock:
        state["forced_exercise"] = exercise
        state["rep_count"]       = 0
        state["hold_time"]       = 0.0
        if exercise is None:
            state["status"]              = "waiting"
            state["exercise"]            = None
            state["feedback_message"]    = ""
            state["form_confidence"]     = 0.0
            state["feedback_confidence"] = 0.0
            state["detection_pct"]       = 0.0

    print(f"  → Forced exercise: {exercise or 'auto'}")
    return jsonify({"ok": True, "exercise": exercise})


# entry point

if __name__ == "__main__":
    print("\n Loading models...")
    models = load_models()

    ai_thread = threading.Thread(target=ai_loop, args=(models,), daemon=True)
    ai_thread.start()

    time.sleep(1)

    print("\n Starting web server...")
    print(" Open your browser at:  http://localhost:8080\n")

    app.run(host="0.0.0.0", port=8080, debug=False, threaded=True)