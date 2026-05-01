import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import glob

# ── KEYPOINT NAME LOOKUP 

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
        axis = parts[2]
        return f"{KP_NAMES.get(kp_key, kp_key)}_{axis}"
    return feature

# ── LOAD DATA ────────────────────────────────────────────────────────────────

csv_files = csv_files = glob.glob("processed/*.csv")
df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)

feature_cols = [
    "left_knee_angle", "right_knee_angle",
    "left_hip_angle", "right_hip_angle",
    "left_elbow_angle", "right_elbow_angle"
]

#  RUN PCA PER EXERCISE 

exercises = ["squat", "lunge", "pushup", "plank"]

for exercise in exercises:
    sub = df[df["exercise"] == exercise].dropna(subset=feature_cols)

    if len(sub) < 10:
        print(f"Skipping {exercise} — not enough data")
        continue

    X = sub[feature_cols].values
    y = sub["mistake"].values

    # Scale first — PCA is sensitive to units
    X_scaled = StandardScaler().fit_transform(X)

    # Reduce to 2 dimensions
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_scaled)

    # ── Print loadings ───────────────────────────────────────────────────────
    loadings = pd.DataFrame(
        pca.components_.T,
        index=[readable_name(f) for f in feature_cols],
        columns=["Component 1", "Component 2"]
    )

    explained = pca.explained_variance_ratio_

    print(f"\n{'='*50}")
    print(f"  {exercise.upper()}")
    print(f"{'='*50}")
    print(f"  Component 1 explains: {explained[0]*100:.1f}% of variance")
    print(f"  Component 2 explains: {explained[1]*100:.1f}% of variance")
    print(f"  Combined:             {sum(explained)*100:.1f}%")
    print(f"\n  Loadings (what each component represents):")
    print(loadings.round(3).sort_values("Component 1", ascending=False).to_string())

    # ── Plot ─────────────────────────────────────────────────────────────────
    plt.figure(figsize=(10, 7))

    for label in sorted(set(y)):
        mask = y == label
        plt.scatter(X_2d[mask, 0], X_2d[mask, 1], label=label, alpha=0.5, s=10)

    plt.legend(loc="best", markerscale=2)
    plt.title(f"PCA — {exercise}\n"
              f"Component 1: {explained[0]*100:.1f}%  |  "
              f"Component 2: {explained[1]*100:.1f}%")
    plt.xlabel(f"Component 1 ({explained[0]*100:.1f}% variance)")
    plt.ylabel(f"Component 2 ({explained[1]*100:.1f}% variance)")
    plt.tight_layout()
    plt.savefig(f"plots/pca_{exercise}.png", dpi=150)
    plt.close()
    print(f"\n  Saved to pca_{exercise}.png")