"""
train_model.py
--------------
GymCoach MVP — Phase 2: Model Training

1. Merges all CSVs from the processed/ folder into one dataset
2. Trains one Random Forest classifier per exercise
3. Each classifier predicts the specific mistake (or "none" for good form)
4. Saves each trained model as a .pkl file in the models/ folder

Usage:
    python train_model.py
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

PROCESSED_DIR = "processed"   # folder containing all per-clip CSVs
MODELS_DIR    = "models"      # output folder for trained .pkl files

# These are the features the model will learn from
# (joint angles + all keypoint x/y coordinates)
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
#  STEP 1 — MERGE ALL CSVs
# ─────────────────────────────────────────────

def merge_csvs():
    """
    Load every CSV from processed/ and combine into one DataFrame.
    Skips transition clips — those are only for exercise detection, not form.
    """
    csv_files = [
        f for f in os.listdir(PROCESSED_DIR)
        if f.endswith(".csv")
    ]

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in '{PROCESSED_DIR}/'. "
            f"Run process_clips.py first."
        )

    print(f"\n Found {len(csv_files)} CSV files in {PROCESSED_DIR}/")

    frames = []
    for f in csv_files:
        path = os.path.join(PROCESSED_DIR, f)
        df   = pd.read_csv(path)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)

    # Drop transition clips — not used for form training
    before = len(combined)
    combined = combined[combined["exercise"] != "transition"]
    dropped  = before - len(combined)

    print(f" Total rows loaded    : {before}")
    print(f" Transition rows dropped : {dropped}")
    print(f" Rows used for training  : {len(combined)}")

    return combined


# ─────────────────────────────────────────────
#  STEP 2 — TRAIN ONE MODEL PER EXERCISE
# ─────────────────────────────────────────────

def train_exercise_model(df_exercise, exercise_name):
    """
    Train a Random Forest on one exercise's data.
    Target: the 'mistake' column (e.g. 'hips_sagging', 'none', 'knees_caving')
    """
    print(f"\n{'─' * 55}")
    print(f"  Training: {exercise_name.upper()}")
    print(f"{'─' * 55}")

    # Check we have all the features we need
    missing_cols = [c for c in ALL_FEATURES if c not in df_exercise.columns]
    if missing_cols:
        print(f"  [!] Missing columns: {missing_cols} — skipping.")
        return None, None

    X = df_exercise[ALL_FEATURES]
    y = df_exercise["mistake"]

    # Show class distribution
    print(f"\n  Class distribution:")
    for mistake, count in y.value_counts().items():
        print(f"    {mistake:<30} {count} frames")

    # Skip if only one class (can't train or evaluate)
    if y.nunique() < 2:
        print(f"\n  [!] Only one class found — need both good and bad form. Skipping.")
        return None, None

    # Encode labels as numbers (Random Forest needs this)
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    # Split into training and test sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    print(f"\n  Training rows : {len(X_train)}")
    print(f"  Test rows     : {len(X_test)}")

    # Train the Random Forest
    model = RandomForestClassifier(
        n_estimators=100,   # number of trees
        max_depth=10,       # prevents overfitting
        random_state=42,
        n_jobs=-1           # use all CPU cores
    )
    model.fit(X_train, y_train)

    # Evaluate on the test set
    y_pred     = model.predict(X_test)
    class_names = encoder.inverse_transform(sorted(set(y_encoded)))

    print(f"\n  Results on test set:")
    print(classification_report(
        y_test, y_pred,
        target_names=class_names,
        zero_division=0
    ))

    return model, encoder


# ─────────────────────────────────────────────
#  STEP 3 — SAVE MODELS
# ─────────────────────────────────────────────

def save_model(model, encoder, exercise_name):
    """
    Save the trained model and its label encoder together
    as a single .pkl file in the models/ folder.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)

    output_path = os.path.join(MODELS_DIR, f"{exercise_name}_model.pkl")

    with open(output_path, "wb") as f:
        pickle.dump({"model": model, "encoder": encoder}, f)

    print(f"  Saved → {output_path}")


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def run():
    # 1. Merge all CSVs
    df = merge_csvs()

    # 2. Get the list of exercises in the data
    exercises = df["exercise"].unique()
    print(f"\n Exercises found: {list(exercises)}")

    trained = []

    # 3. Train one model per exercise
    for exercise in sorted(exercises):
        df_ex = df[df["exercise"] == exercise]

        model, encoder = train_exercise_model(df_ex, exercise)

        if model is not None:
            save_model(model, encoder, exercise)
            trained.append(exercise)

    # 4. Summary
    print(f"\n{'─' * 55}")
    print(f" Training complete!")
    print(f" Models saved for: {trained}")
    print(f" Location: {MODELS_DIR}/")
    print(f"{'─' * 55}\n")


if __name__ == "__main__":
    run()
