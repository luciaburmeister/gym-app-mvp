# GymCoach MVP

An AI-powered real-time exercise form feedback system. Stand in front of your webcam, perform an exercise, and receive live coaching on your form.

Built with YOLOv8-pose for pose estimation and scikit-learn Random Forest classifiers for form analysis.

---

## What it does

- **Automatically detects** which exercise you are performing (squat, push-up, lunge, or plank)
- **Analyses your form** in real time and flags specific mistakes (e.g. "Knees caving in — push them out")
- **Counts your reps** for squat, push-up, and lunge; tracks hold time for plank
- **Web interface** accessible from any browser on your local network — no app install needed

---

## Supported exercises

| Exercise | Form mistakes detected |
|----------|----------------------|
| Squat | Knees caving, chest dropping, half rep, heels rising |
| Push-up | Elbows flaring, head dropping, hips raised |
| Lunge | Knee past toes, torso leaning, not 90°, uneven hips |
| Plank | Hips sagging, hips too high, head position, shoulders forward |

---

## Project structure

```
gym-app-mvp/
│
├── app.py                      # Main Flask server — run this to use the app
├── train_model.py              # Trains one form classifier per exercise
├── train_exercise_detector.py  # Trains the exercise detector model
├── process_clips.py            # Batch processes raw video clips into CSVs
├── feature_importance.py       # Visualises which features matter most
├── fix_labels.py               # Utility to clean and fix label files
├── pca.py                      # PCA analysis on pose features
│
├── labels/                     # Processed training CSVs (committed to repo)
│   ├── idle.csv
│   ├── lunge.csv
│   ├── plank.csv
│   ├── pushups.csv
│   └── squat.csv
│
├── templates/
│   └── index.html              # Browser UI
│
├── models/                     # Trained .pkl files — git-ignored, train locally
├── plots/                      # Training charts output — git-ignored
│
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/luciaburmeister/gym-app-mvp.git
cd gym-app-mvp
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # Mac / Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the models

The `.pkl` model files are not stored in the repo. Train them locally using the label CSVs already included in the repo — no raw video needed.

**Step 1 — Train the exercise detector** (identifies which exercise you are doing):

```bash
python train_exercise_detector.py
```

**Step 2 — Train the form classifiers** (one model per exercise, identifies specific mistakes):

```bash
python train_model.py
```

Both scripts save their output to `models/` automatically.

### 5. Run the app

```bash
python app.py
```

Then open your browser at **http://localhost:8080**

---

## How the pipeline works

```
Webcam frame
    │
    ▼
YOLOv8-pose
    │  17 body keypoints (x, y) + confidence per keypoint
    ▼
Feature extraction
    │  8 joint angles + 17 keypoint positions = 42 features
    │  + 9 asymmetry features + torso angle   = 52 features (detector only)
    ▼
Exercise detector (Random Forest)
    │  Classifies: squat / pushup / lunge / plank / transition
    │  Requires 70% agreement over 20 consecutive frames to confirm
    ▼
Form classifier (Random Forest, one per exercise)
    │  Classifies: good_form / specific mistake
    ▼
Feedback displayed in browser + rep counted
```

---

## If you have new training data

Only needed if you have recorded new video clips and want to retrain from scratch.

### Step 1 — Process your raw clips

Raw clips should be organised by label and placed in `raw_clips/`. Then run:

```bash
python process_clips.py
```

This generates CSVs in `processed/` which can then be merged into the `labels/` folder.

### Step 2 — Retrain

Follow the same training steps as above (train_exercise_detector.py → train_model.py).

---

## Tech stack

| Layer | Technology |
|-------|-----------|
| Pose estimation | YOLOv8-pose (Ultralytics) |
| Form classification | scikit-learn Random Forest |
| Backend | Python, Flask |
| Frontend | HTML, CSS, Vanilla JS |
| Video streaming | MJPEG over HTTP |

---

## Known limitations (MVP scope)

- Accuracy depends heavily on training data volume — more clips = better feedback
- Works best when the full body is visible in frame and the camera is at roughly chest height
- Single person only — multi-person support not implemented
- Trained on a limited set of body types and camera angles

---

