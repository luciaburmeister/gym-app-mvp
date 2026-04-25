import pickle
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import glob
import os

MODELS_DIR = "/Users/lucia/Desktop/GymCoach/models"
OUTPUT_DIR = "/Users/lucia/Desktop/GymCoach"

ANGLE_FEATURES = [
    "left_knee_angle", "right_knee_angle",
    "left_hip_angle", "right_hip_angle",
    "left_elbow_angle", "right_elbow_angle",
    "left_shoulder_angle", "right_shoulder_angle"
]

KP_NAMES = {
    "kp_0":  "nose",
    "kp_1":  "left_eye",
    "kp_2":  "right_eye",
    "kp_3":  "left_ear",
    "kp_4":  "right_ear",
    "kp_5":  "left_shoulder",
    "kp_6":  "right_shoulder",
    "kp_7":  "left_elbow",
    "kp_8":  "right_elbow",
    "kp_9":  "left_wrist",
    "kp_10": "right_wrist",
    "kp_11": "left_hip",
    "kp_12": "right_hip",
    "kp_13": "left_knee",
    "kp_14": "right_knee",
    "kp_15": "left_ankle",
    "kp_16": "right_ankle",
}

def readable_name(feature):
    parts = feature.split("_")
    if parts[0] == "kp":
        kp_key = f"kp_{parts[1]}"
        axis = parts[2]  # x or y
        return f"{KP_NAMES.get(kp_key, kp_key)}_{axis}"
    return feature  # angle features stay as-is

model_files = {
    "squat":  "squat_model.pkl",
    "lunge":  "lunge_model.pkl",
    "pushup": "pushup_model.pkl",
    "plank":  "plank_model.pkl",
}

for exercise, filename in model_files.items():
    path = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(path):
        print(f"Skipping {exercise} — model file not found at {path}")
        continue

    with open(path, "rb") as f:
        obj = pickle.load(f)

    model    = obj["model"]
    encoder  = obj["encoder"]
    features = list(model.feature_names_in_)
    importances = model.feature_importances_

    # Build dataframe and sort
    df = pd.DataFrame({"feature": features, "importance": importances})
    df = df.sort_values("importance", ascending=False).head(15)
    df["is_angle"] = df["feature"].isin(ANGLE_FEATURES)
    df["display_name"] = df["feature"].apply(readable_name)

    # Summary stats
    angle_total = sum(importances[i] for i, f in enumerate(features) if f in ANGLE_FEATURES)
    kp_total    = sum(importances[i] for i, f in enumerate(features) if f not in ANGLE_FEATURES)

    print(f"\n{'='*50}")
    print(f"  {exercise.upper()}")
    print(f"{'='*50}")
    print(f"  Classes: {list(encoder.classes_)}")
    print(f"  Angle features total importance:    {angle_total*100:.1f}%")
    print(f"  Keypoint features total importance: {kp_total*100:.1f}%")
    print(f"\n  Top 15 features:")
    for _, row in df.iterrows():
        tag = " <-- angle" if row["is_angle"] else ""
        name = row["display_name"]
        print(f"    {name:<30} {row['importance']*100:.2f}%{tag}")

    # Plot with readable names
    colors = ["#111111" if is_angle else "#BBBBBB" for is_angle in df["is_angle"]]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(df["display_name"][::-1], df["importance"][::-1], color=colors[::-1])

    ax.set_xlabel("Importance score")
    ax.set_title(f"Feature importance — {exercise}\nClasses: {', '.join(encoder.classes_)}", pad=12)

    angle_patch = mpatches.Patch(color="#111111", label="Angle feature")
    kp_patch    = mpatches.Patch(color="#BBBBBB", label="Keypoint position")
    ax.legend(handles=[angle_patch, kp_patch], loc="lower right")

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, f"feature_importance_{exercise}.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\n  Saved chart to {out_path}")

# ── CROSS-EXERCISE COMPARISON (angles only) ──────────────────────────────────

angle_data = {}

for exercise, filename in model_files.items():
    path = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(path):
        continue
    with open(path, "rb") as f:
        obj = pickle.load(f)
    model = obj["model"]
    features = list(model.feature_names_in_)
    importances = model.feature_importances_
    angle_data[exercise] = {
        f: importances[features.index(f)]
        for f in ANGLE_FEATURES
        if f in features
    }

df_cross = pd.DataFrame(angle_data).T  # rows = exercises, cols = angles

fig, ax = plt.subplots(figsize=(12, 5))
x = range(len(ANGLE_FEATURES))
bar_width = 0.2
colors = ["#111111", "#555555", "#999999", "#CCCCCC"]

for i, (exercise, color) in enumerate(zip(df_cross.index, colors)):
    offsets = [xi + i * bar_width for xi in x]
    ax.bar(offsets, df_cross.loc[exercise, ANGLE_FEATURES],
           width=bar_width, label=exercise, color=color)

ax.set_xticks([xi + bar_width * 1.5 for xi in x])
ax.set_xticklabels([f.replace("_angle", "").replace("_", "\n") for f in ANGLE_FEATURES], fontsize=9)
ax.set_ylabel("Importance score")
ax.set_title("Angle feature importance — compared across exercises")
ax.legend()

plt.tight_layout()
out_path = os.path.join(OUTPUT_DIR, "feature_importance_comparison.png")
plt.savefig(out_path, dpi=150)
plt.close()
print(f"\nSaved cross-exercise comparison to {out_path}")