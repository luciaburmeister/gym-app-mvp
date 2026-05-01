"""
train_model.py
--------------
GymCoach MVP — Form Classifier Training

1. Merges all CSVs from the processed/ folder into one dataset
2. Normalizes keypoints so position/height don't affect the model
3. Trains one Random Forest classifier per exercise
4. Each classifier predicts the specific mistake (or "good_form")
5. Saves each trained model as a .pkl file in the models/ folder
6. Generates training/validation curves, baseline comparison, error analysis

v3 changes:
- Keypoint normalization added: keypoints translated to hip origin and
  scaled by torso length, making the model position and scale invariant.
  This means the model works correctly regardless of where in the frame
  the person stands or how tall they are.

Usage:
    python train_model.py
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    train_test_split, learning_curve, StratifiedKFold, GridSearchCV
)
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    balanced_accuracy_score
)
from sklearn.preprocessing import LabelEncoder
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    print("\n  [!] imbalanced-learn not installed.")
    print("      Run: pip install imbalanced-learn")
    print("      SMOTE oversampling will be skipped.\n")


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

MIN_SAMPLES_PER_CLASS = 20
CONFIDENCE_THRESHOLD  = 0.55
CV_FOLDS              = 5
TUNE_HYPERPARAMS      = True

PARAM_GRID = {
    "n_estimators":      [100, 200],
    "max_depth":         [8, 12, None],
    "min_samples_split": [2, 5],
}


# ─────────────────────────────────────────────
#  KEYPOINT NORMALIZATION
#  Must match normalize_keypoints() in app.py exactly
# ─────────────────────────────────────────────

def normalize_keypoints(df):
    """
    Makes keypoint coordinates position and scale invariant.

    Without this, the model confuses "person standing on the left of frame"
    with "person standing on the right" — they look like different poses
    because the x/y pixel values are completely different.

    Steps:
      1. Translate: subtract hip midpoint so hips are always at (0, 0)
      2. Scale: divide by torso length (hip-to-shoulder distance) so
         tall and short people have comparable coordinate ranges

    This must stay in sync with normalize_keypoints() in app.py.
    """
    df = df.copy()

    hip_mid_x      = (df["kp_11_x"] + df["kp_12_x"]) / 2
    hip_mid_y      = (df["kp_11_y"] + df["kp_12_y"]) / 2
    shoulder_mid_x = (df["kp_5_x"]  + df["kp_6_x"])  / 2
    shoulder_mid_y = (df["kp_5_y"]  + df["kp_6_y"])  / 2

    torso_length = np.sqrt(
        (shoulder_mid_x - hip_mid_x) ** 2 +
        (shoulder_mid_y - hip_mid_y) ** 2
    ) + 1e-6   # avoid division by zero

    for i in range(17):
        df[f"kp_{i}_x"] = (df[f"kp_{i}_x"] - hip_mid_x) / torso_length
        df[f"kp_{i}_y"] = (df[f"kp_{i}_y"] - hip_mid_y) / torso_length

    return df


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

    print(f"\n  Found {len(csv_files)} CSV file(s) in {PROCESSED_DIR}/")

    frames = []
    for f in csv_files:
        path = os.path.join(PROCESSED_DIR, f)
        df = pd.read_csv(path)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)

    before   = len(combined)
    combined = combined[combined["exercise"] != "transition"]
    dropped  = before - len(combined)

    print(f"  Total rows loaded       : {before}")
    print(f"  Transition rows dropped : {dropped}")
    print(f"  Rows used for training  : {len(combined)}")

    return combined


# ─────────────────────────────────────────────
#  HELPER — CLASS BALANCE DIAGNOSIS
# ─────────────────────────────────────────────

def diagnose_class_balance(y_series, exercise_name):
    counts   = y_series.value_counts()
    total    = len(y_series)
    majority = counts.iloc[0]

    print(f"\n  Class distribution for {exercise_name}:")
    warnings_found = []

    for cls, count in counts.items():
        pct  = count / total * 100
        bar  = "█" * int(pct / 2)
        flag = ""

        if count < MIN_SAMPLES_PER_CLASS:
            flag = "  ⚠️  TOO FEW SAMPLES"
            warnings_found.append(cls)
        elif count / majority < 0.2:
            flag = "  ⚠️  SEVERELY IMBALANCED"
            warnings_found.append(cls)

        print(f"    {cls:<30} {count:>5} frames  ({pct:4.1f}%)  {bar}{flag}")

    if warnings_found:
        print(f"\n  ⚠️  Classes with data problems: {warnings_found}")
        print(f"     → Record more clips for these mistake types.")
        print(f"     → SMOTE will try to compensate but more real data is better.")

    return warnings_found


# ─────────────────────────────────────────────
#  HELPER — SMOTE
# ─────────────────────────────────────────────

def apply_smote(X_train, y_train):
    if not SMOTE_AVAILABLE:
        return X_train, y_train

    counts    = pd.Series(y_train).value_counts()
    min_count = counts.min()
    k         = max(1, min_count - 1) if min_count < 6 else 5

    try:
        smote        = SMOTE(random_state=42, k_neighbors=k)
        X_res, y_res = smote.fit_resample(X_train, y_train)
        added        = len(X_res) - len(X_train)
        print(f"\n  SMOTE: generated {added} synthetic samples to balance classes")
        return X_res, y_res
    except Exception as e:
        print(f"\n  [!] SMOTE failed: {e} — training without oversampling")
        return X_train, y_train


# ─────────────────────────────────────────────
#  HELPER — HYPERPARAMETER TUNING
# ─────────────────────────────────────────────

def tune_hyperparameters(X_train, y_train):
    print(f"\n  Tuning hyperparameters (this may take a minute)...")

    base_model = RandomForestClassifier(
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        base_model, PARAM_GRID,
        cv=cv,
        scoring="balanced_accuracy",
        n_jobs=-1,
        verbose=0
    )

    grid_search.fit(X_train, y_train)

    print(f"  Best params  : {grid_search.best_params_}")
    print(f"  Best CV score: {grid_search.best_score_*100:.1f}% balanced accuracy")

    return grid_search.best_estimator_


# ─────────────────────────────────────────────
#  PLOTS
# ─────────────────────────────────────────────

def plot_learning_curve(model, X, y, exercise_name, class_names):
    print(f"  Generating learning curve...")

    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y,
        cv=CV_FOLDS,
        n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring="balanced_accuracy"
    )

    train_mean = train_scores.mean(axis=1)
    train_std  = train_scores.std(axis=1)
    val_mean   = val_scores.mean(axis=1)
    val_std    = val_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(train_sizes, train_mean, label="Training score",   color="#111111")
    ax.plot(train_sizes, val_mean,   label="Validation score", color="#888888", linestyle="--")
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                    alpha=0.1, color="#111111")
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                    alpha=0.1, color="#888888")

    ax.set_xlabel("Training set size")
    ax.set_ylabel("Balanced Accuracy")
    ax.set_title(f"Learning curve — {exercise_name}\nClasses: {', '.join(class_names)}")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"learning_curve_{exercise_name}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved → {path}")


def plot_confusion_matrix(y_test, y_pred, class_names, exercise_name):
    cm      = confusion_matrix(y_test, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Greys",
                xticklabels=class_names, yticklabels=class_names, ax=axes[0])
    axes[0].set_title(f"Confusion matrix — {exercise_name}\n(raw counts)")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")

    sns.heatmap(cm_norm, annot=True, fmt=".0%", cmap="Greys",
                xticklabels=class_names, yticklabels=class_names, ax=axes[1])
    axes[1].set_title(f"Confusion matrix — {exercise_name}\n(% of actual class)")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"confusion_matrix_{exercise_name}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved → {path}")


def baseline_comparison(X_train, X_test, y_train, y_test, rf_model, class_names, exercise_name):
    results = {}
    results["Random Forest (yours)"]       = balanced_accuracy_score(y_test, rf_model.predict(X_test))

    dummy_freq = DummyClassifier(strategy="most_frequent")
    dummy_freq.fit(X_train, y_train)
    results["Baseline: always most common"] = balanced_accuracy_score(y_test, dummy_freq.predict(X_test))

    dummy_strat = DummyClassifier(strategy="stratified", random_state=42)
    dummy_strat.fit(X_train, y_train)
    results["Baseline: random guess"]       = balanced_accuracy_score(y_test, dummy_strat.predict(X_test))

    print(f"\n  Baseline comparison (balanced accuracy):")
    for name, acc in results.items():
        bar = "█" * int(acc * 40)
        print(f"    {name:<40} {acc*100:.1f}%  {bar}")

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#111111", "#AAAAAA", "#CCCCCC"]
    bars   = ax.barh(list(results.keys()), list(results.values()), color=colors)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Balanced Accuracy")
    ax.set_title(f"Baseline comparison — {exercise_name}")

    for bar, val in zip(bars, results.values()):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val*100:.1f}%", va="center", fontsize=10)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"baseline_comparison_{exercise_name}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved → {path}")

    return results


def plot_feature_importance(model, exercise_name, top_n=15):
    importances  = model.feature_importances_
    indices      = np.argsort(importances)[::-1][:top_n]
    top_features = [ALL_FEATURES[i] for i in indices]
    top_values   = importances[indices]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(top_features[::-1], top_values[::-1], color="#333333")
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_n} features — {exercise_name}")
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"feature_importance_{exercise_name}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved → {path}")


# ─────────────────────────────────────────────
#  STEP 2 — TRAIN ONE MODEL PER EXERCISE
# ─────────────────────────────────────────────

def train_exercise_model(df_exercise, exercise_name):
    print(f"\n{'─' * 60}")
    print(f"  Training: {exercise_name.upper()}")
    print(f"{'─' * 60}")

    missing_cols = [c for c in ALL_FEATURES if c not in df_exercise.columns]
    if missing_cols:
        print(f"  [!] Missing columns: {missing_cols} — skipping.")
        return None, None

    # Normalize keypoints before extracting features
    df_exercise = normalize_keypoints(df_exercise)

    X = df_exercise[ALL_FEATURES].copy()
    y = df_exercise["mistake"].copy()

    valid_mask = X.notna().any(axis=1)
    X = X[valid_mask]
    y = y[valid_mask]

    X = X.fillna(X.median())

    problem_classes = diagnose_class_balance(y, exercise_name)

    if y.nunique() < 2:
        print(f"\n  [!] Only one class found — need at least good form + one mistake.")
        return None, None

    encoder     = LabelEncoder()
    y_encoded   = encoder.fit_transform(y)
    class_names = list(encoder.classes_)

    print(f"\n  Classes: {class_names}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded
    )

    print(f"\n  Training rows : {len(X_train)}")
    print(f"  Test rows     : {len(X_test)}")

    X_train_res, y_train_res = apply_smote(X_train, y_train)

    if TUNE_HYPERPARAMS:
        model = tune_hyperparameters(X_train_res, y_train_res)
    else:
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_split=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train_res, y_train_res)

    y_pred = model.predict(X_test)

    print(f"\n  Results on test set:")
    print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))

    bal_acc = balanced_accuracy_score(y_test, y_pred)
    print(f"  Balanced accuracy: {bal_acc*100:.1f}%")
    print(f"  (Balanced accuracy weights all classes equally — better metric than plain accuracy)")

    report = classification_report(y_test, y_pred, target_names=class_names,
                                   zero_division=0, output_dict=True)
    for cls in class_names:
        if cls in report and report[cls]["recall"] < 0.5:
            print(f"\n  ⚠️  WARNING: '{cls}' recall is only {report[cls]['recall']*100:.0f}%")
            print(f"     The model is missing most real '{cls}' cases.")
            print(f"     → Record more '{cls}' clips to fix this.")

    os.makedirs(PLOTS_DIR, exist_ok=True)

    plot_learning_curve(model, X, y_encoded, exercise_name, class_names)
    plot_confusion_matrix(y_test, y_pred, class_names, exercise_name)
    baseline_comparison(X_train, X_test, y_train, y_test, model, class_names, exercise_name)
    plot_feature_importance(model, exercise_name)

    return model, encoder


# ─────────────────────────────────────────────
#  STEP 3 — SAVE MODELS
# ─────────────────────────────────────────────

def save_model(model, encoder, exercise_name):
    os.makedirs(MODELS_DIR, exist_ok=True)
    output_path = os.path.join(MODELS_DIR, f"{exercise_name}_model.pkl")
    payload = {
        "model":                model,
        "encoder":              encoder,
        "features":             ALL_FEATURES,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "class_names":          list(encoder.classes_),
        "normalized":           True,   # flag so app.py knows to normalize at inference
    }
    with open(output_path, "wb") as f:
        pickle.dump(payload, f)
    print(f"  Saved → {output_path}")


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def run():
    print("\n" + "═" * 60)
    print("  GymCoach MVP — Model Training (v3 — with normalization)")
    print("═" * 60)

    df = merge_csvs()

    exercises = df["exercise"].unique()
    print(f"\n  Exercises found: {list(exercises)}")

    trained = []

    for exercise in sorted(exercises):
        df_ex          = df[df["exercise"] == exercise]
        model, encoder = train_exercise_model(df_ex, exercise)

        if model is not None:
            save_model(model, encoder, exercise)
            trained.append(exercise)

    print(f"\n{'═' * 60}")
    print(f"  Training complete!")
    print(f"  Models saved for : {trained}")
    print(f"  Models location  : {MODELS_DIR}/")
    print(f"  Plots location   : {PLOTS_DIR}/")
    print(f"{'═' * 60}\n")


if __name__ == "__main__":
    run()