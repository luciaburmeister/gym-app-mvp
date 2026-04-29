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
├── agent/
│   ├── app.py                      # Main Flask server — run this to use the app
│   ├── train_model.py              # Trains one form classifier per exercise
│   ├── train_exercise_detector.py  # Trains the exercise detector model
│   ├── feature_importance.py       # Visualises which features matter most
│   └── live.py                     # Standalone webcam version (no browser needed)
│
├── web/
│   └── templates/
│       └── index.html              # Browser UI
│
├── models/                         # Trained .pkl files (git-ignored)
├── processed/                      # Processed CSV data (git-ignored)
├── plots/                          # Training charts output (git-ignored)
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

### 4. Get the trained models

The `.pkl` model files are not stored in the repo (they are large binary files). Get them from the shared Google Drive folder and place them in `agent/models/`:

```
agent/models/
├── exercise_detector.pkl
├── squat_model.pkl
├── pushup_model.pkl
├── lunge_model.pkl
└── plank_model.pkl
```

### 5. Run the app

```bash
cd agent
python app.py
```

Then open your browser at **http://localhost:8080**

---

## Training your own models

Only needed if you have recorded new training data and want to retrain.

### Step 1 — Process your video clips

Your raw clips should be labelled and processed into CSVs using `process_clips.py` (see Google Drive for the script and raw data). Processed CSVs go into `agent/processed/`.

### Step 2 — Train the exercise detector

This model decides *which* exercise you are doing.

```bash
cd agent
python train_exercise_detector.py
```

### Step 3 — Train the form classifiers

One model per exercise, each deciding *what mistake* you are making.

```bash
python train_model.py
```

Both scripts save their output to `agent/models/` automatically.

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

## Team

IE University — built as part of an applied AI project.
