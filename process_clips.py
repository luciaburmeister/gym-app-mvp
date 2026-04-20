"""
process_clips.py
----------------
GymCoach MVP — Phase 1: Video Processing

Runs YOLOv8-pose on every filmed clip and extracts joint angles
and keypoint coordinates frame by frame. Outputs one CSV per clip,
then merges everything into a single training_data.csv.

Usage:
    python process_clips.py

Expected folder structure:
    gym_coach/
        raw_clips/
            pushups/
                male_pushups_side_good_form_01.mov
                ...
            squat/
            lunge/
            plank/
        labels/
            pushups.csv
            squat.csv
            lunge.csv
            plank.csv
"""

import cv2
import math
import os
import pandas as pd
from ultralytics import YOLO


# ─────────────────────────────────────────────
#  CONFIG — edit these paths if needed
# ─────────────────────────────────────────────

RAW_CLIPS_DIR  = "raw_clips"          # folder containing all filmed clips
LABELS_DIR     = "labels"             # folder containing one CSV per exercise
PROCESSED_DIR  = "processed"          # output folder for per-clip CSVs
OUTPUT_CSV     = "training_data.csv"  # final merged file used for training
MODEL_WEIGHTS  = "yolov8n-pose.pt"    # downloads automatically on first run

# Minimum confidence to consider a joint visible (0.0 - 1.0)
CONFIDENCE_THRESHOLD = 0.5

# YOLO keypoint indices (17 joints)
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
#  GEOMETRY HELPERS
# ─────────────────────────────────────────────

def get_angle(a, b, c):
    """
    Calculate the angle (in degrees) at joint b,
    given three 2D points a, b, c.

    Example: get_angle(hip, knee, ankle) → knee bend angle
    """
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])

    dot_product = ba[0] * bc[0] + ba[1] * bc[1]
    magnitude   = (ba[0]**2 + ba[1]**2)**0.5 * (bc[0]**2 + bc[1]**2)**0.5

    if magnitude == 0:
        return 0.0

    # Clamp to [-1, 1] to avoid floating point errors in acos
    angle = math.degrees(math.acos(max(-1.0, min(1.0, dot_product / magnitude))))
    return round(angle, 2)


def calculate_angles(kp):
    """
    Given a (17, 2) array of keypoint coordinates,
    return a dictionary of all relevant joint angles.
    """
    return {
        "left_knee_angle":      get_angle(kp[LEFT_HIP],       kp[LEFT_KNEE],   kp[LEFT_ANKLE]),
        "right_knee_angle":     get_angle(kp[RIGHT_HIP],      kp[RIGHT_KNEE],  kp[RIGHT_ANKLE]),
        "left_hip_angle":       get_angle(kp[LEFT_SHOULDER],  kp[LEFT_HIP],    kp[LEFT_KNEE]),
        "right_hip_angle":      get_angle(kp[RIGHT_SHOULDER], kp[RIGHT_HIP],   kp[RIGHT_KNEE]),
        "left_elbow_angle":     get_angle(kp[LEFT_SHOULDER],  kp[LEFT_ELBOW],  kp[LEFT_WRIST]),
        "right_elbow_angle":    get_angle(kp[RIGHT_SHOULDER], kp[RIGHT_ELBOW], kp[RIGHT_WRIST]),
        "left_shoulder_angle":  get_angle(kp[LEFT_HIP],       kp[LEFT_SHOULDER],  kp[LEFT_ELBOW]),
        "right_shoulder_angle": get_angle(kp[RIGHT_HIP],      kp[RIGHT_SHOULDER], kp[RIGHT_ELBOW]),
    }


# ─────────────────────────────────────────────
#  CORE PROCESSING
# ─────────────────────────────────────────────

def process_clip(model, video_path, exercise, label, mistake, output_csv):
    """
    Process a single video clip:
      - Run YOLO on every frame
      - Extract joint coordinates and calculate angles
      - Save results to a CSV file
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"  [!] Could not open video: {video_path}")
        return 0

    rows        = []
    frame_index = 0
    skipped     = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_index += 1
        results = model(frame, verbose=False)

        # Skip frame if YOLO found no person
        if not results or results[0].keypoints is None:
            skipped += 1
            continue

        if len(results[0].keypoints.xy) == 0:
            skipped += 1
            continue

        kp   = results[0].keypoints.xy[0].cpu().numpy()    # shape: (17, 2)
        conf = results[0].keypoints.conf[0].cpu().numpy()  # shape: (17,)

        # Skip frame if hips are not visible — needed for all exercises
        if conf[LEFT_HIP] < CONFIDENCE_THRESHOLD or conf[RIGHT_HIP] < CONFIDENCE_THRESHOLD:
            skipped += 1
            continue

        # Build the row: metadata + angles + all 17 raw keypoints
        row = {
            "exercise": exercise,
            "label":    label,
            "mistake":  mistake,
            "frame":    frame_index,
            **calculate_angles(kp),
            **{f"kp_{i}_x":    round(float(kp[i][0]),   2) for i in range(17)},
            **{f"kp_{i}_y":    round(float(kp[i][1]),   2) for i in range(17)},
            **{f"kp_{i}_conf": round(float(conf[i]),    3) for i in range(17)},
        }
        rows.append(row)

    cap.release()

    if rows:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        pd.DataFrame(rows).to_csv(output_csv, index=False)

    print(f"  ✓  {os.path.basename(video_path):<45} "
          f"{len(rows)} frames saved  |  {skipped} skipped")

    return len(rows)


# ─────────────────────────────────────────────
#  LABEL LOADING
# ─────────────────────────────────────────────

def load_all_labels():
    """
    Read every CSV from the labels/ folder and combine them
    into a single DataFrame. Each CSV should have at minimum:
    filename, exercise, label, mistake
    """
    if not os.path.exists(LABELS_DIR):
        raise FileNotFoundError(
            f"labels/ folder not found. "
            f"Please create it and add one CSV per exercise (e.g. pushups.csv, squat.csv)."
        )

    csv_files = [f for f in os.listdir(LABELS_DIR) if f.endswith(".csv")]

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in '{LABELS_DIR}/'. "
            f"Please export each exercise sheet as a CSV and place it there."
        )

    print(f"\n Found {len(csv_files)} label file(s): {', '.join(csv_files)}")

    frames = []
    for csv_file in csv_files:
        path = os.path.join(LABELS_DIR, csv_file)
        df   = pd.read_csv(path)

        required_columns = {"filename", "exercise", "label", "mistake"}
        if not required_columns.issubset(df.columns):
            raise ValueError(
                f"'{csv_file}' is missing required columns. "
                f"Required: {required_columns}. Found: {set(df.columns)}"
            )

        frames.append(df)
        print(f"   ✓  {csv_file:<20} {len(df)} clips")

    combined = pd.concat(frames, ignore_index=True)
    print(f"\n   Total clips across all labels: {len(combined)}")
    return combined


# ─────────────────────────────────────────────
#  BATCH RUNNER
# ─────────────────────────────────────────────

def run_all_clips():
    """
    Load all label CSVs from labels/, find each video file,
    process it, then merge all outputs into training_data.csv.
    """
    labels_df = load_all_labels()

    # Load YOLO model (downloads weights on first run)
    print("\n Loading YOLOv8-pose model...")
    model = YOLO(MODEL_WEIGHTS)
    print(" Model ready.\n")

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    total_clips  = len(labels_df)
    total_frames = 0
    all_csvs     = []

    print(f"{'─' * 65}")
    print(f" Processing {total_clips} clips...")
    print(f"{'─' * 65}")

    for _, row in labels_df.iterrows():
        filename = row["filename"]
        exercise = row["exercise"].strip().lower()
        label    = row["label"].strip().lower()
        mistake  = str(row.get("mistake", "")).strip().lower()

        # Find the video file inside raw_clips/
        video_path = find_video(filename)

        if video_path is None:
            print(f"  [!] File not found, skipping: {filename}")
            continue

        # Output CSV goes into processed/ with the same base name
        base_name  = os.path.splitext(filename)[0]
        output_csv = os.path.join(PROCESSED_DIR, f"{base_name}.csv")

        frames_saved = process_clip(
            model, video_path, exercise, label, mistake, output_csv
        )

        if frames_saved > 0:
            all_csvs.append(output_csv)
            total_frames += frames_saved

    # Merge all per-clip CSVs into one master file
    print(f"\n{'─' * 65}")
    print(f" Merging {len(all_csvs)} CSV files into {OUTPUT_CSV}...")

    if all_csvs:
        merged = pd.concat([pd.read_csv(f) for f in all_csvs], ignore_index=True)
        merged.to_csv(OUTPUT_CSV, index=False)

        print(f"\n Done!")
        print(f"   Total clips processed : {len(all_csvs)}")
        print(f"   Total frames saved    : {total_frames}")
        print(f"   Output file           : {OUTPUT_CSV}")
        print(f"   Shape                 : {merged.shape[0]} rows × {merged.shape[1]} columns")
        print(f"\n Label breakdown:")
        print(merged.groupby(["exercise", "mistake"])["frame"].count().to_string())
    else:
        print(" [!] No clips were processed successfully. Check your labels/ folder and raw_clips/.")

    print(f"{'─' * 65}\n")


# ─────────────────────────────────────────────
#  UTILITIES
# ─────────────────────────────────────────────

def find_video(filename):
    """
    Search for a video file anywhere inside RAW_CLIPS_DIR.
    Returns the full path if found, None otherwise.
    """
    for root, _, files in os.walk(RAW_CLIPS_DIR):
        if filename in files:
            return os.path.join(root, filename)
    return None


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    run_all_clips()
