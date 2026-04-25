"""
train_model.py
--------------
GymCoach MVP — Phase 2: Model Training

1. Merges all CSVs from the processed/ folder into one dataset
2. Trains one Random Forest classifier per exercise
3. Each classifier predicts the specific mistake (or "none" for good form)
4. Saves each trained model as a .pkl file in the models/ folder
5. Generates training/validation curves, baseline comparison, error analysis

Usage:
    python train_model.py
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.dummy import DummyClassifier


# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────

PROCESSED_DIR = "processed"
MODELS_DIR    = "models"
PLOTS_DIR     = "plots"

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
    csv_files = [f for f in os.listdir(PROCESSED_DIR) if f.endswith(".csv")]

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in '{PROCESSED_DIR}/'. "
            f"Run process_clips.py first."
        )

    print(f"\n Found {len(csv_files)} CSV files in {PROCESSED_DIR}/")

    frames = []
    for f in csv_files:
        path = os.path.join(PROCESSED_DIR, f)
        frames.append(pd.read_csv(path))

    combined = pd.concat(frames, ignore_index=True)

    before   = len(combined)
    combined = combined[combined["exercise"] != "transition"]
    dropped  = before - len(combined)

    print(f" Total rows loaded       : {before}")
    print(f" Transition rows dropped : {dropped}")
    print(f" Rows used for training  : {len(combined)}")

    return combined


# ─────────────────────────────────────────────
#  PLOT 1 — TRAINING / VALIDATION CURVE
# ─────────────────────────────────────────────

def plot_learning_curve(model, X, y, exercise_name, class_names):
    """
    Shows how model accuracy changes as more training data is added.
    If training accuracy is high but validation is low → overfitting.
    If both are low → underfitting / not enough data.
    """
    print(f"  Generating learning curve...")

    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y,
        cv=5,
        n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring="accuracy"
    )

    train_mean = train_scores.mean(axis=1)
    train_std  = train_scores.std(axis=1)
    val_mean   = val_scores.mean(axis=1)
    val_std    = val_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(train_sizes, train_mean, color="#111111", label="Training accuracy")
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                    alpha=0.1, color="#111111")

    ax.plot(train_sizes, val_mean, color="#888888", linestyle="--",
            label="Validation accuracy (cross-val)")
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                    alpha=0.1, color="#888888")

    ax.set_xlabel("Training set size")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Learning curve — {exercise_name}\nClasses: {', '.join(class_names)}")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"learning_curve_{exercise_name}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved learning curve → {path}")


# ─────────────────────────────────────────────
#  PLOT 2 — CONFUSION MATRIX (error analysis)
# ─────────────────────────────────────────────

def plot_confusion_matrix(y_test, y_pred, class_names, exercise_name):
    """
    Shows which mistakes the model confuses with each other.
    Dark diagonal = correct predictions.
    Off-diagonal = errors.
    """
    cm = confusion_matrix(y_test, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Raw counts
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greys",
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[0])
    axes[0].set_title(f"Confusion matrix — {exercise_name}\n(raw counts)")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")

    # Normalised (percentages)
    sns.heatmap(cm_norm, annot=True, fmt=".0%", cmap="Greys",
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[1])
    axes[1].set_title(f"Confusion matrix — {exercise_name}\n(% of actual class)")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"confusion_matrix_{exercise_name}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved confusion matrix → {path}")


# ─────────────────────────────────────────────
#  PLOT 3 — BASELINE COMPARISON
# ─────────────────────────────────────────────

def baseline_comparison(X_train, X_test, y_train, y_test, rf_model, class_names, exercise_name):
    """
    Compares your Random Forest against two dumb baselines:
    - Most frequent: always predicts the most common class
    - Stratified:    predicts randomly based on class proportions
    Your model should beat both by a clear margin.
    """
    results = {}

    # Random Forest
    results["Random Forest"] = accuracy_score(y_test, rf_model.predict(X_test))

    # Baseline 1: always predict the most common class
    dummy_freq = DummyClassifier(strategy="most_frequent")
    dummy_freq.fit(X_train, y_train)
    results["Baseline: most frequent"] = accuracy_score(y_test, dummy_freq.predict(X_test))

    # Baseline 2: predict randomly but proportionally
    dummy_strat = DummyClassifier(strategy="stratified", random_state=42)
    dummy_strat.fit(X_train, y_train)
    results["Baseline: random"] = accuracy_score(y_test, dummy_strat.predict(X_test))

    print(f"\n  Baseline comparison:")
    for name, acc in results.items():
        bar = "█" * int(acc * 40)
        print(f"    {name:<35} {acc*100:.1f}%  {bar}")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#111111", "#AAAAAA", "#CCCCCC"]
    bars = ax.barh(list(results.keys()), list(results.values()), color=colors)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Accuracy")
    ax.set_title(f"Baseline comparison — {exercise_name}")

    for bar, val in zip(bars, results.values()):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val*100:.1f}%", va="center", fontsize=10)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"baseline_comparison_{exercise_name}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved baseline comparison → {path}")

    return results


# ─────────────────────────────────────────────
#  STEP 2 — TRAIN ONE MODEL PER EXERCISE
# ─────────────────────────────────────────────

def train_exercise_model(df_exercise, exercise_name):
    print(f"\n{'─' * 55}")
    print(f"  Training: {exercise_name.upper()}")
    print(f"{'─' * 55}")

    missing_cols = [c for c in ALL_FEATURES if c not in df_exercise.columns]
    if missing_cols:
        print(f"  [!] Missing columns: {missing_cols} — skipping.")
        return None, None

    X = df_exercise[ALL_FEATURES]
    y = df_exercise["mistake"]

    print(f"\n  Class distribution:")
    for mistake, count in y.value_counts().items():
        print(f"    {mistake:<30} {count} frames")

    if y.nunique() < 2:
        print(f"\n  [!] Only one class found — skipping.")
        return None, None

    encoder   = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    class_names = list(encoder.classes_)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    print(f"\n  Training rows : {len(X_train)}")
    print(f"  Test rows     : {len(X_test)}")

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print(f"\n  Results on test set:")
    print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))

    # ── Analysis plots ───────────────────────────────────────────────────────
    os.makedirs(PLOTS_DIR, exist_ok=True)

    plot_learning_curve(model, X, y_encoded, exercise_name, class_names)
    plot_confusion_matrix(y_test, y_pred, class_names, exercise_name)
    baseline_comparison(X_train, X_test, y_train, y_test, model, class_names, exercise_name)

    return model, encoder


# ─────────────────────────────────────────────
#  STEP 3 — SAVE MODELS
# ─────────────────────────────────────────────

def save_model(model, encoder, exercise_name):
    os.makedirs(MODELS_DIR, exist_ok=True)
    output_path = os.path.join(MODELS_DIR, f"{exercise_name}_model.pkl")
    with open(output_path, "wb") as f:
        pickle.dump({"model": model, "encoder": encoder}, f)
    print(f"  Saved → {output_path}")


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def run():
    df = merge_csvs()

    exercises = df["exercise"].unique()
    print(f"\n Exercises found: {list(exercises)}")

    trained = []

    for exercise in sorted(exercises):
        df_ex          = df[df["exercise"] == exercise]
        model, encoder = train_exercise_model(df_ex, exercise)

        if model is not None:
            save_model(model, encoder, exercise)
            trained.append(exercise)

    print(f"\n{'─' * 55}")
    print(f" Training complete!")
    print(f" Models saved for : {trained}")
    print(f" Models location  : {MODELS_DIR}/")
    print(f" Plots location   : {PLOTS_DIR}/")
    print(f"{'─' * 55}\n")


if __name__ == "__main__":
    run()