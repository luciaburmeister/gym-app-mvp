"""
train_exercise_detector.py
--------------------------
GymCoach MVP — Train the Exercise Auto-Detector

Trains a Random Forest classifier that looks at pose features
and predicts WHICH exercise the person is performing.

Classes: squat, lunge, plank, pushup, transition (idle/walking)

This model runs first in the live pipeline — once it confidently
identifies the exercise, the matching form classifier is loaded.

Usage:
    python train_exercise_detector.py
"""

import os
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder


# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────

PROCESSED_DIR = "processed"
MODELS_DIR    = "models"
OUTPUT_FILE   = "exercise_detector.pkl"

ANGLE_FEATURES = [
    "left_knee_angle",
    "right_knee_angle",
    "left_hip_angle",
    "right_hip_angle",
    "left_elbow_angle",
    "right_elbow_angle",
    "left_shoulder_angle",
    "right_shoulder_angle",
]

KEYPOINT_FEATURES = (
    [f"kp_{i}_x" for i in range(17)] +
    [f"kp_{i}_y" for i in range(17)]
)

ALL_FEATURES = ANGLE_FEATURES + KEYPOINT_FEATURES


# ─────────────────────────────────────────────
#  LOAD & PREPARE DATA
# ─────────────────────────────────────────────

def load_data():
    csv_files = [f for f in os.listdir(PROCESSED_DIR) if f.endswith(".csv")]

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in '{PROCESSED_DIR}/'. "
            "Run process_clips.py first."
        )

    print(f"\n Found {len(csv_files)} CSV files in {PROCESSED_DIR}/")

    frames = []
    for f in csv_files:
        df = pd.read_csv(os.path.join(PROCESSED_DIR, f))
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)

    # Normalise exercise names (just in case)
    combined["exercise"] = combined["exercise"].str.strip().str.lower()

    # Collapse all transition variants to one label
    combined["exercise"] = combined["exercise"].replace({
        "transition": "transition"
    })

    print(f" Total rows          : {len(combined)}")
    print(f"\n Class distribution (exercise):")
    for exercise, count in combined["exercise"].value_counts().items():
        print(f"   {exercise:<15} {count} frames")

    return combined


# ─────────────────────────────────────────────
#  TRAIN
# ─────────────────────────────────────────────

def train(df):
    print(f"\n{'─' * 55}")
    print(f"  Training: Exercise Detector")
    print(f"{'─' * 55}")

    X = df[ALL_FEATURES]
    y = df["exercise"]

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    print(f"\n  Classes: {list(encoder.classes_)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded
    )

    print(f"  Training rows : {len(X_train)}")
    print(f"  Test rows     : {len(X_test)}")

    model = RandomForestClassifier(
        n_estimators=150,  # slightly more trees for a harder task
        max_depth=12,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print(f"\n  Results on test set:")
    print(classification_report(
        y_test, y_pred,
        target_names=encoder.classes_,
        zero_division=0
    ))

    return model, encoder


# ─────────────────────────────────────────────
#  SAVE
# ─────────────────────────────────────────────

def save(model, encoder):
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, OUTPUT_FILE)
    with open(path, "wb") as f:
        pickle.dump({"model": model, "encoder": encoder}, f)
    print(f"\n  Saved → {path}")


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def run():
    df            = load_data()
    model, encoder = train(df)
    save(model, encoder)

    print(f"\n{'─' * 55}")
    print(f" Done! Exercise detector saved to {MODELS_DIR}/{OUTPUT_FILE}")
    print(f" Now run:  python3 live.py")
    print(f"{'─' * 55}\n")


if __name__ == "__main__":
    run()
