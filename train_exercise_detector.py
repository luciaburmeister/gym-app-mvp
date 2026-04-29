"""
train_exercise_detector.py
--------------------------
GymCoach MVP — Exercise Detector Training

This is a SEPARATE script from train_model.py.
train_model.py trains one form classifier per exercise (good vs mistakes).
This script trains the exercise detector — the model that decides
WHICH exercise you are doing (squat, pushup, lunge, plank, or transition).

Run this BEFORE running app.py.

FIX v2:
- Feature vector now matches app.py exactly: 42 base + 9 asymmetry + 1 torso = 52 features
- Added torso_angle: best single feature for upright (squat/lunge) vs horizontal (plank/pushup)
- Added 9 asymmetry features: left/right differences that strongly distinguish lunge from squat

Usage:
    python train_exercise_detector.py
"""

import os
import pickle
import warnings
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    print("\n  [!] imbalanced-learn not installed — SMOTE skipped.")
    print("      Run: pip install imbalanced-learn\n")


#configuration constants

PROCESSED_DIR = "processed"
MODELS_DIR    = "models"
PLOTS_DIR     = "plots"

# Keypoint indices (YOLO ordering)
LEFT_SHOULDER  = 5;  RIGHT_SHOULDER = 6
LEFT_ELBOW     = 7;  RIGHT_ELBOW    = 8
LEFT_WRIST     = 9;  RIGHT_WRIST    = 10
LEFT_HIP       = 11; RIGHT_HIP      = 12
LEFT_KNEE      = 13; RIGHT_KNEE     = 14
LEFT_ANKLE     = 15; RIGHT_ANKLE    = 16

# Base angle + keypoint features (same as form classifiers)
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

# Extra features only the detector uses
# These are computed from the base keypoints at training time
ASYMMETRY_FEATURES = [
    "asym_knee_angle",
    "asym_hip_angle",
    "asym_shoulder_angle",
    "asym_elbow_angle",
    "asym_hip_x",
    "asym_knee_x",
    "asym_ankle_x",
    "asym_knee_y",
    "asym_ankle_y",
]

EXTRA_FEATURES = ASYMMETRY_FEATURES + ["torso_angle"]

BASE_FEATURES   = ANGLE_FEATURES + KEYPOINT_FEATURES        # 42 features
DETECTOR_FEATURES = BASE_FEATURES + EXTRA_FEATURES          # 52 features

TUNE_HYPERPARAMS = True

PARAM_GRID = {
    "n_estimators":      [100, 200, 300],
    "max_depth":         [10, 15, None],
    "min_samples_split": [2, 5],
}


# feature engineering 
#  Must match extract_features_detector() in app.py exactly


def add_detector_features(df):
    """
    Takes a dataframe with the 42 base features and adds:
    - 9 asymmetry features (left vs right differences)
    - 1 torso angle (upright vs horizontal — best discriminator for exercise type)

    This function must stay in sync with extract_features_detector() in app.py.
    """

    #Asymmetry features 
    df["asym_knee_angle"]     = (df["left_knee_angle"]     - df["right_knee_angle"]).abs()
    df["asym_hip_angle"]      = (df["left_hip_angle"]      - df["right_hip_angle"]).abs()
    df["asym_shoulder_angle"] = (df["left_shoulder_angle"] - df["right_shoulder_angle"]).abs()
    df["asym_elbow_angle"]    = (df["left_elbow_angle"]    - df["right_elbow_angle"]).abs()

    df["asym_hip_x"]          = (df["kp_11_x"] - df["kp_12_x"]).abs()
    df["asym_knee_x"]         = (df["kp_13_x"] - df["kp_14_x"]).abs()
    df["asym_ankle_x"]        = (df["kp_15_x"] - df["kp_16_x"]).abs()
    df["asym_knee_y"]         = (df["kp_13_y"] - df["kp_14_y"]).abs()
    df["asym_ankle_y"]        = (df["kp_15_y"] - df["kp_16_y"]).abs()

    # torso angle 
    # Angle of the line from hip midpoint to shoulder midpoint.
    # 90° = standing upright (squat, lunge)
    # 0°  = horizontal (plank, pushup)
    #most powerful angles to detect differences 
    shoulder_mid_x = (df["kp_5_x"]  + df["kp_6_x"])  / 2
    shoulder_mid_y = (df["kp_5_y"]  + df["kp_6_y"])  / 2
    hip_mid_x      = (df["kp_11_x"] + df["kp_12_x"]) / 2
    hip_mid_y      = (df["kp_11_y"] + df["kp_12_y"]) / 2

    dx = shoulder_mid_x - hip_mid_x
    dy = shoulder_mid_y - hip_mid_y

    # atan2 gives angle in radians; convert to degrees
    # We take the absolute value so both left/right lean map to the same value
    df["torso_angle"] = np.degrees(np.arctan2(dy.abs(), dx.abs() + 1e-6))

    return df


#load data - step 1

def load_data():
    csv_files = [f for f in os.listdir(PROCESSED_DIR) if f.endswith(".csv")]

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in '{PROCESSED_DIR}/'. "
            "Run process_clips.py first."
        )

    print(f"\n  Found {len(csv_files)} CSV files in {PROCESSED_DIR}/")

    frames = []
    for f in csv_files:
        path = os.path.join(PROCESSED_DIR, f)
        df   = pd.read_csv(path)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    total    = len(combined)

    print(f"\n  Total rows loaded: {total:,}")
    print(f"\n  Exercise distribution (this is what the detector will learn):")

    counts = combined["exercise"].value_counts()
    for ex, count in counts.items():
        pct = count / total * 100
        bar = "█" * int(pct / 2)
        print(f"    {ex:<15} {count:>7,} frames  ({pct:4.1f}%)  {bar}")

    return combined


#prepare labels and features - step 2

def prepare(df):
    # Check base features exist
    missing = [c for c in BASE_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in data: {missing}")

    # Add asymmetry + torso features
    print(f"\n  Adding asymmetry and torso angle features...")
    df = add_detector_features(df)

    X = df[DETECTOR_FEATURES].copy()
    y = df["exercise"].copy()

    # Drop rows where all features are NaN
    valid = X.notna().any(axis=1)
    X     = X[valid]
    y     = y[valid]

    # Fill remaining NaNs with column median
    X = X.fillna(X.median())

    print(f"  Feature vector size: {X.shape[1]} features per frame")
    print(f"  Features: {BASE_FEATURES[:4]}... + {len(EXTRA_FEATURES)} extra (asymmetry + torso)")

    return X, y


#apply smote - step 3 

def apply_smote(X_train, y_train):
    if not SMOTE_AVAILABLE:
        return X_train, y_train

    counts    = pd.Series(y_train).value_counts()
    min_count = counts.min()
    k         = max(1, min(5, min_count - 1))

    try:
        smote        = SMOTE(random_state=42, k_neighbors=k)
        X_res, y_res = smote.fit_resample(X_train, y_train)
        added        = len(X_res) - len(X_train)
        print(f"\n  SMOTE: generated {added:,} synthetic samples to balance classes")
        return X_res, y_res
    except Exception as e:
        print(f"\n  [!] SMOTE failed: {e} — training without oversampling")
        return X_train, y_train


#train - step 4

def train(X_train, y_train):
    if TUNE_HYPERPARAMS:
        print(f"\n  Tuning hyperparameters (this may take a minute)...")
        base = RandomForestClassifier(
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
        cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        search = GridSearchCV(
            base, PARAM_GRID,
            cv=cv,
            scoring="balanced_accuracy",
            n_jobs=-1,
            verbose=0
        )
        search.fit(X_train, y_train)
        print(f"  Best params  : {search.best_params_}")
        print(f"  Best CV score: {search.best_score_*100:.1f}% balanced accuracy")
        return search.best_estimator_
    else:
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        return model


#evaluate - step 5

def evaluate(model, encoder, X_test, y_test):
    y_pred      = model.predict(X_test)
    class_names = list(encoder.classes_)

    print(f"\n  Results on held-out test set:")
    print(classification_report(y_test, y_pred,
                                target_names=class_names, zero_division=0))

    bal_acc = balanced_accuracy_score(y_test, y_pred)
    print(f"  Balanced accuracy: {bal_acc*100:.1f}%")

    # Per-class recall warning
    report = classification_report(y_test, y_pred,
                                   target_names=class_names,
                                   zero_division=0, output_dict=True)
    for cls in class_names:
        if cls in report and report[cls]["recall"] < 0.7:
            print(f"\n  ⚠️  WARNING: '{cls}' recall is only "
                  f"{report[cls]['recall']*100:.0f}%")
            print(f"     The detector is struggling to recognise {cls}.")
            print(f"     → Record more {cls} clips to fix this.")

    # Confusion matrix
    os.makedirs(PLOTS_DIR, exist_ok=True)
    cm      = confusion_matrix(y_test, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greys",
                xticklabels=class_names, yticklabels=class_names, ax=axes[0])
    axes[0].set_title("Exercise Detector — Confusion Matrix (raw counts)")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")

    sns.heatmap(cm_norm, annot=True, fmt=".0%", cmap="Greys",
                xticklabels=class_names, yticklabels=class_names, ax=axes[1])
    axes[1].set_title("Exercise Detector — Confusion Matrix (% of actual)")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "confusion_matrix_exercise_detector.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\n  Saved confusion matrix → {path}")

    # Feature importance
    importances  = model.feature_importances_
    indices      = np.argsort(importances)[::-1][:15]
    top_features = [DETECTOR_FEATURES[i] for i in indices]
    top_values   = importances[indices]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(top_features[::-1], top_values[::-1], color="#333333")
    ax.set_xlabel("Importance")
    ax.set_title("Exercise Detector — Top 15 Features")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "feature_importance_exercise_detector.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved feature importance → {path}")


#save - step 6

def save(model, encoder):
    os.makedirs(MODELS_DIR, exist_ok=True)
    path    = os.path.join(MODELS_DIR, "exercise_detector.pkl")
    payload = {
        "model":       model,
        "encoder":     encoder,
        "features":    DETECTOR_FEATURES,   # saved so app.py can verify at load time
        "class_names": list(encoder.classes_),
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    print(f"\n  Saved → {path}")


#main

def run():
    print("\n" + "═" * 60)
    print("  GymCoach MVP — Exercise Detector Training (v2)")
    print("═" * 60)

    df       = load_data()
    X, y_raw = prepare(df)

    encoder   = LabelEncoder()
    y_encoded = encoder.fit_transform(y_raw)
    classes   = list(encoder.classes_)
    print(f"\n  Classes the detector will learn: {classes}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded
    )
    print(f"\n  Training rows : {len(X_train):,}")
    print(f"  Test rows     : {len(X_test):,}")

    X_train, y_train = apply_smote(X_train, y_train)

    model = train(X_train, y_train)

    evaluate(model, encoder, X_test, y_test)

    save(model, encoder)

    print(f"\n{'═' * 60}")
    print(f"  Done! exercise_detector.pkl saved to {MODELS_DIR}/")
    print(f"  You can now run:  python app.py")
    print(f"{'═' * 60}\n")


if __name__ == "__main__":
    run()