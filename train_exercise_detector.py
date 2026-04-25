"""
train_exercise_detector.py
--------------------------
GymCoach MVP — Train the Exercise Auto-Detector (v2)

Adds asymmetry features to help distinguish squat vs lunge:
  - In a SQUAT  both sides are roughly equal  → low asymmetry
  - In a LUNGE  one leg is forward, one back  → high asymmetry

New features added:
  - knee_asymmetry        = abs(left_knee_angle  - right_knee_angle)
  - hip_asymmetry         = abs(left_hip_angle   - right_hip_angle)
  - shoulder_asymmetry    = abs(left_shoulder_angle - right_shoulder_angle)
  - elbow_asymmetry       = abs(left_elbow_angle - right_elbow_angle)
  - lateral_hip_offset    = abs(kp_11_x - kp_12_x)
  - lateral_knee_offset   = abs(kp_13_x - kp_14_x)
  - lateral_ankle_offset  = abs(kp_15_x - kp_16_x)
  - vertical_knee_diff    = abs(kp_13_y - kp_14_y)
  - vertical_ankle_diff   = abs(kp_15_y - kp_16_y)

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

ASYMMETRY_FEATURES = [
    "knee_asymmetry",
    "hip_asymmetry",
    "shoulder_asymmetry",
    "elbow_asymmetry",
    "lateral_hip_offset",
    "lateral_knee_offset",
    "lateral_ankle_offset",
    "vertical_knee_diff",
    "vertical_ankle_diff",
]

ALL_FEATURES = ANGLE_FEATURES + KEYPOINT_FEATURES + ASYMMETRY_FEATURES


# ─────────────────────────────────────────────
#  ASYMMETRY FEATURES
# ─────────────────────────────────────────────

def add_asymmetry_features(df):
    df = df.copy()
    df["knee_asymmetry"]      = (df["left_knee_angle"]     - df["right_knee_angle"]).abs()
    df["hip_asymmetry"]       = (df["left_hip_angle"]      - df["right_hip_angle"]).abs()
    df["shoulder_asymmetry"]  = (df["left_shoulder_angle"] - df["right_shoulder_angle"]).abs()
    df["elbow_asymmetry"]     = (df["left_elbow_angle"]    - df["right_elbow_angle"]).abs()
    df["lateral_hip_offset"]  = (df["kp_11_x"] - df["kp_12_x"]).abs()
    df["lateral_knee_offset"] = (df["kp_13_x"] - df["kp_14_x"]).abs()
    df["lateral_ankle_offset"]= (df["kp_15_x"] - df["kp_16_x"]).abs()
    df["vertical_knee_diff"]  = (df["kp_13_y"] - df["kp_14_y"]).abs()
    df["vertical_ankle_diff"] = (df["kp_15_y"] - df["kp_16_y"]).abs()
    return df


# ─────────────────────────────────────────────
#  LOAD DATA
# ─────────────────────────────────────────────

def load_data():
    csv_files = [f for f in os.listdir(PROCESSED_DIR) if f.endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in '{PROCESSED_DIR}/'.")

    print(f"\n Found {len(csv_files)} CSV files in {PROCESSED_DIR}/")

    frames = []
    for f in csv_files:
        frames.append(pd.read_csv(os.path.join(PROCESSED_DIR, f)))

    combined = pd.concat(frames, ignore_index=True)
    combined["exercise"] = combined["exercise"].str.strip().str.lower()
    combined = add_asymmetry_features(combined)

    print(f" Total rows: {len(combined)}")
    print(f"\n Class distribution:")
    for exercise, count in combined["exercise"].value_counts().items():
        print(f"   {exercise:<15} {count} frames")

    print(f"\n Average knee asymmetry per exercise (squat should be low, lunge high):")
    for exercise, grp in combined.groupby("exercise"):
        avg = grp["knee_asymmetry"].mean()
        bar = "█" * int(avg / 5)
        print(f"   {exercise:<15} {avg:5.1f}°  {bar}")

    return combined


# ─────────────────────────────────────────────
#  TRAIN
# ─────────────────────────────────────────────

def train(df):
    print(f"\n{'─' * 55}")
    print(f"  Training: Exercise Detector (v2 — with asymmetry)")
    print(f"{'─' * 55}")

    X = df[ALL_FEATURES]
    y = df["exercise"]

    encoder   = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    print(f"\n  Classes        : {list(encoder.classes_)}")
    print(f"  Total features : {len(ALL_FEATURES)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    print(f"  Training rows  : {len(X_train)}")
    print(f"  Test rows      : {len(X_test)}")

    model = RandomForestClassifier(
        n_estimators=150,
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

    # Top features
    feat_imp = sorted(
        zip(ALL_FEATURES, model.feature_importances_),
        key=lambda x: x[1], reverse=True
    )[:10]
    print(f"  Top 10 features:")
    for feat, imp in feat_imp:
        bar = "█" * int(imp * 200)
        print(f"    {feat:<30} {imp:.4f}  {bar}")

    return model, encoder


# ─────────────────────────────────────────────
#  SAVE
# ─────────────────────────────────────────────

def save(model, encoder):
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, OUTPUT_FILE)
    with open(path, "wb") as f:
        pickle.dump({
            "model":    model,
            "encoder":  encoder,
            "features": ALL_FEATURES,
        }, f)
    print(f"\n  Saved → {path}")


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def run():
    df             = load_data()
    model, encoder = train(df)
    save(model, encoder)

    print(f"\n{'─' * 55}")
    print(f" Done! Now restart app.py.")
    print(f"{'─' * 55}\n")


if __name__ == "__main__":
    run()
