"""
process_missing.py
------------------
GymCoach MVP — Process only the missing clips that were skipped.
Run this after process_clips.py to fill in the gaps.

Usage:
    python process_missing.py
"""

import cv2
import math
import os
import pandas as pd
from ultralytics import YOLO


# ─────────────────────────────────────────────
#  CONFIG — same as process_clips.py
# ─────────────────────────────────────────────

RAW_CLIPS_DIR        = "raw_clips"
PROCESSED_DIR        = "processed"
OUTPUT_CSV           = "training_data.csv"
MODEL_WEIGHTS        = "yolov8n-pose.pt"
CONFIDENCE_THRESHOLD = 0.5

LEFT_SHOULDER  = 5
RIGHT_SHOULDER = 6
LEFT_ELBOW     = 7
RIGHT_ELBOW    = 8
LEFT_WRIST     = 9
RIGHT_WRIST    = 10
LEFT_HIP       = 11
RIGHT_HIP      = 12
LEFT_KNEE      = 13
RIGHT_KNEE     = 14
LEFT_ANKLE     = 15
RIGHT_ANKLE    = 16


# ─────────────────────────────────────────────
#  MISSING CLIPS — copied from your error log
# ─────────────────────────────────────────────
#
#  Format: (filename, exercise, label, mistake)
#  Edit this list if you add or remove clips.
#
MISSING_CLIPS = [
    # Plank
    ("male_plank_side_head_01.mp4",              "plank",    "bad",  "head_position"),
    ("male_plank_side_hips_high_01.mp4",         "plank",    "bad",  "hips_high"),
    ("male_plank_side_hips_sagging_01.mp4",      "plank",    "bad",  "hips_sagging"),

    # Squat — bad form
    ("male_squat_front_knees_caving_01.mov",     "squat",    "bad",  "knees_caving"),
    ("male_squat_front_knees_caving_02.mov",     "squat",    "bad",  "knees_caving"),
    ("male_squat_front_knees_caving_03.mov",     "squat",    "bad",  "knees_caving"),
    ("male_squat_side_heels_up_01.mov",          "squat",    "bad",  "heels_up"),
    ("male_squat_side_heels_up_02.mov",          "squat",    "bad",  "heels_up"),
    ("male_squat_side_heels_up_03.mov",          "squat",    "bad",  "heels_up"),
    ("male_squat_side_half_rep_01.mov",          "squat",    "bad",  "half_rep"),
    ("male_squat_side_half_rep_02.mov",          "squat",    "bad",  "half_rep"),
    ("male_squat_side_half_rep_03.mov",          "squat",    "bad",  "half_rep"),
    ("male_squat_side_chest_down_01.mov",        "squat",    "bad",  "chest_down"),
    ("male_squat_side_chest_down_02.mov",        "squat",    "bad",  "chest_down"),
    ("male_squat_side_chest_down_03.mov",        "squat",    "bad",  "chest_down"),

    # Squat — good form
    ("male_squat_front_good_form_01.mov",        "squat",    "good", "none"),
    ("male_squat_front_good_form_02.mov",        "squat",    "good", "none"),
    ("male_squat_front_good_form_03.mov",        "squat",    "good", "none"),
    ("male_squat_side_good_form_01.mov",         "squat",    "good", "none"),
    ("male_squat_side_good_form_02.mov",         "squat",    "good", "none"),
    ("male_squat_side_good_form_03.mov",         "squat",    "good", "none"),

    # Lunge — good form
    ("female_lunge_side_goodform_01.mov",        "lunge",    "good", "none"),
    ("female_lunge_side_goodform_02.mov",        "lunge",    "good", "none"),
    ("female_lunge_side_goodform_03.mov",        "lunge",    "good", "none"),

    # Lunge — bad form
    ("female_lunge_side_kneepasttoes_01.mov",    "lunge",    "bad",  "knee_past_toes"),
    ("female_lunge_side_kneepasttoes_02.mov",    "lunge",    "bad",  "knee_past_toes"),
    ("female_lunge_side_torsoleaning_01.mov",    "lunge",    "bad",  "torso_leaning"),
    ("female_lunge_side_torsoleaning_02.mov",    "lunge",    "bad",  "torso_leaning"),
    ("female_lunge_side_90degrees_01.mov",       "lunge",    "bad",  "not_90_degrees"),
    ("female_lunge_side_90degrees_02.mov",       "lunge",    "bad",  "not_90_degrees"),
    ("female_lunge_front_uneven_01.mov",         "lunge",    "bad",  "uneven_hips"),
    ("female_lunge_front_uneven_02.mov",         "lunge",    "bad",  "uneven_hips"),

    # Transitions
    ("male_transition_squat_side_01.mov",        "transition", "idle", "none"),
    ("male_transition_squat_side_02.mov",        "transition", "idle", "none"),
    ("male_transition_squat_side_03.mov",        "transition", "idle", "none"),
    ("male_transition_squat_side_04.mov",        "transition", "idle", "none"),
    ("male_transition_squat_side_05.mov",        "transition", "idle", "none"),

    # Push-ups — bad form
    ("female_pushups_side_hips_raised_01.mov",      "pushup",   "bad",  "hips_raised"),
    ("female_pushups_side_hips_raised_02.mov",      "pushup",   "bad",  "hips_raised"),
    ("female_pushups_side_head_dropping_01.mov",    "pushup",   "bad",  "head_dropping"),
    ("female_pushups_side_head_dropping_02.mov",    "pushup",   "bad",  "head_dropping"),
    ("female_pushups_side_elbows_flaring_01.mov",   "pushup",   "bad",  "elbows_flaring"),
    ("female_pushups_side_elbows_flaring_02.mov",   "pushup",   "bad",  "elbows_flaring"),

    # Push-ups — good form
    ("female_pushups_side_good_form_01.mov",        "pushup",   "good", "none"),
    ("female_pushups_side_good_form_02.mov",        "pushup",   "good", "none"),
    ("female_pushups_side_good_form_03.mov",        "pushup",   "good", "none"),
]


# ─────────────────────────────────────────────
#  GEOMETRY HELPERS (same as process_clips.py)
# ─────────────────────────────────────────────

def get_angle(a, b, c):
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])
    dot_product = ba[0] * bc[0] + ba[1] * bc[1]
    magnitude   = (ba[0]**2 + ba[1]**2)**0.5 * (bc[0]**2 + bc[1]**2)**0.5
    if magnitude == 0:
        return 0.0
    angle = math.degrees(math.acos(max(-1.0, min(1.0, dot_product / magnitude))))
    return round(angle, 2)


def calculate_angles(kp):
    return {
        "left_knee_angle":      get_angle(kp[LEFT_HIP],       kp[LEFT_KNEE],      kp[LEFT_ANKLE]),
        "right_knee_angle":     get_angle(kp[RIGHT_HIP],      kp[RIGHT_KNEE],     kp[RIGHT_ANKLE]),
        "left_hip_angle":       get_angle(kp[LEFT_SHOULDER],  kp[LEFT_HIP],       kp[LEFT_KNEE]),
        "right_hip_angle":      get_angle(kp[RIGHT_SHOULDER], kp[RIGHT_HIP],      kp[RIGHT_KNEE]),
        "left_elbow_angle":     get_angle(kp[LEFT_SHOULDER],  kp[LEFT_ELBOW],     kp[LEFT_WRIST]),
        "right_elbow_angle":    get_angle(kp[RIGHT_SHOULDER], kp[RIGHT_ELBOW],    kp[RIGHT_WRIST]),
        "left_shoulder_angle":  get_angle(kp[LEFT_HIP],       kp[LEFT_SHOULDER],  kp[LEFT_ELBOW]),
        "right_shoulder_angle": get_angle(kp[RIGHT_HIP],      kp[RIGHT_SHOULDER], kp[RIGHT_ELBOW]),
    }


# ─────────────────────────────────────────────
#  CORE PROCESSING
# ─────────────────────────────────────────────

def find_video(filename):
    for root, _, files in os.walk(RAW_CLIPS_DIR):
        if filename in files:
            return os.path.join(root, filename)
    return None


def process_clip(model, video_path, exercise, label, mistake, output_csv):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"  [!] Could not open: {video_path}")
        return 0

    rows, frame_index, skipped = [], 0, 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_index += 1
        results = model(frame, verbose=False)

        if not results or results[0].keypoints is None:
            skipped += 1
            continue
        if len(results[0].keypoints.xy) == 0:
            skipped += 1
            continue

        kp   = results[0].keypoints.xy[0].cpu().numpy()
        conf = results[0].keypoints.conf[0].cpu().numpy()

        if conf[LEFT_HIP] < CONFIDENCE_THRESHOLD or conf[RIGHT_HIP] < CONFIDENCE_THRESHOLD:
            skipped += 1
            continue

        row = {
            "exercise": exercise,
            "label":    label,
            "mistake":  mistake,
            "frame":    frame_index,
            **calculate_angles(kp),
            **{f"kp_{i}_x":    round(float(kp[i][0]), 2) for i in range(17)},
            **{f"kp_{i}_y":    round(float(kp[i][1]), 2) for i in range(17)},
            **{f"kp_{i}_conf": round(float(conf[i]),  3) for i in range(17)},
        }
        rows.append(row)

    cap.release()

    if rows:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        pd.DataFrame(rows).to_csv(output_csv, index=False)

    print(f"  ✓  {os.path.basename(video_path):<50} "
          f"{len(rows)} frames  |  {skipped} skipped")
    return len(rows)


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def run_missing():
    print(f"\n Loading YOLOv8-pose model...")
    model = YOLO(MODEL_WEIGHTS)
    print(f" Model ready.\n")
    print(f"{'─' * 65}")
    print(f" Processing {len(MISSING_CLIPS)} missing clips...")
    print(f"{'─' * 65}\n")

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    new_csvs     = []
    total_frames = 0
    not_found    = []

    for filename, exercise, label, mistake in MISSING_CLIPS:
        video_path = find_video(filename)

        if video_path is None:
            print(f"  [!] Still not found, skipping: {filename}")
            not_found.append(filename)
            continue

        base_name  = os.path.splitext(filename)[0]
        output_csv = os.path.join(PROCESSED_DIR, f"{base_name}.csv")

        frames_saved = process_clip(model, video_path, exercise, label, mistake, output_csv)

        if frames_saved > 0:
            new_csvs.append(output_csv)
            total_frames += frames_saved

    # ── Merge new CSVs into existing training_data.csv ──
    print(f"\n{'─' * 65}")

    if new_csvs:
        new_data = pd.concat([pd.read_csv(f) for f in new_csvs], ignore_index=True)

        if os.path.exists(OUTPUT_CSV):
            existing = pd.read_csv(OUTPUT_CSV)
            merged   = pd.concat([existing, new_data], ignore_index=True)
            print(f" Appending to existing {OUTPUT_CSV}...")
        else:
            merged = new_data
            print(f" Creating new {OUTPUT_CSV}...")

        merged.to_csv(OUTPUT_CSV, index=False)

        print(f"\n Done!")
        print(f"   New clips processed  : {len(new_csvs)}")
        print(f"   New frames added     : {total_frames}")
        print(f"   Total rows in CSV    : {len(merged)}")
        print(f"\n Label breakdown:")
        print(merged.groupby(["exercise", "mistake"])["frame"].count().to_string())
    else:
        print(" [!] No new clips were processed.")

    if not_found:
        print(f"\n Still missing ({len(not_found)} files) — check raw_clips/ folder:")
        for f in not_found:
            print(f"   - {f}")

    print(f"{'─' * 65}\n")


if __name__ == "__main__":
    run_missing()
