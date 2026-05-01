"""
fix_labels.py
-------------
GymCoach MVP — Fix inconsistent label names across all processed CSVs.

Problems this fixes:
  - 'pushups' → 'pushup'  (duplicate exercise name)
  - 'good form' / 'none'  → 'good_form'
  - 'knees caving'        → 'knees_caving'
  - 'chest down'          → 'chest_down'
  - 'half rep'            → 'half_rep'
  - 'heels up'            → 'heels_up'
  - 'head'                → 'head_position'
  - 'head up'             → 'head_up'
  - 'torso leaning forward' / 'torso_leaning' → 'torso_leaning'
  - 'knees going past toes' / 'knee_past_toes' → 'knee_past_toes'
  - 'knees not reaching 90 degrees' / 'not_90_degrees' → 'not_90_degrees'
  - 'uneven weight distribution' / 'uneven_hips' → 'uneven_hips'
  - 'hips raised' / 'hips sagging' → 'hips_raised' / 'hips_sagging'
  - 'head dropping' → 'head_dropping'
  - 'elbows flaring' → 'elbows_flaring'
  - 'good_form' label column → kept as 'good_form'

Usage:
    python fix_labels.py
"""

import os
import pandas as pd


PROCESSED_DIR = "processed"

#maps 

EXERCISE_MAP = {
    "pushups": "pushup",
}

MISTAKE_MAP = {
    # Good form / no mistake
    "none":                           "good_form",
    "good form":                      "good_form",
    "idle":                           "good_form",

    # Squat
    "knees caving":                   "knees_caving",
    "chest down":                     "chest_down",
    "half rep":                       "half_rep",
    "heels up":                       "heels_up",

    # Plank
    "head":                           "head_position",
    "head up":                        "head_up",
    "shoulders forward":              "shoulders_forward",
    "hips sagging":                   "hips_sagging",
    "hips high":                      "hips_high",

    # Lunge
    "torso leaning forward":          "torso_leaning",
    "knees going past toes":          "knee_past_toes",
    "knees not reaching 90 degrees":  "not_90_degrees",
    "uneven weight distribution":     "uneven_hips",

    # Push-up
    "hips raised":                    "hips_raised",
    "head dropping":                  "head_dropping",
    "elbows flaring":                 "elbows_flaring",
}

# label column (good/bad) — normalise these too
LABEL_MAP = {
    "good": "good",
    "bad":  "bad",
    "idle": "good",
}


# main 

def fix_all():
    csv_files = [f for f in os.listdir(PROCESSED_DIR) if f.endswith(".csv")]

    if not csv_files:
        print(f"[!] No CSV files found in {PROCESSED_DIR}/")
        return

    print(f"\n Fixing labels in {len(csv_files)} CSV files...\n")

    total_exercise_fixes = 0
    total_mistake_fixes  = 0

    for filename in sorted(csv_files):
        path = os.path.join(PROCESSED_DIR, filename)
        df   = pd.read_csv(path)

        # Fix exercise column
        ex_before = df["exercise"].copy()
        df["exercise"] = df["exercise"].str.strip().str.lower().replace(EXERCISE_MAP)
        ex_changes = (ex_before != df["exercise"]).sum()
        total_exercise_fixes += ex_changes

        # Fix label column
        df["label"] = df["label"].str.strip().str.lower().replace(LABEL_MAP)

        # Fix mistake column
        mk_before = df["mistake"].copy()
        df["mistake"] = df["mistake"].str.strip().str.lower().replace(MISTAKE_MAP)
        mk_changes = (mk_before != df["mistake"]).sum()
        total_mistake_fixes += mk_changes

        df.to_csv(path, index=False)

        if ex_changes > 0 or mk_changes > 0:
            print(f"  ✓  {filename:<55} "
                  f"exercise fixes: {ex_changes}  |  mistake fixes: {mk_changes}")

    print(f"\n Done!")
    print(f"   Total exercise fixes : {total_exercise_fixes}")
    print(f"   Total mistake fixes  : {total_mistake_fixes}")
    print(f"\n Now run:  python3 train_model.py")
    print(f" to retrain with the cleaned labels.\n")


if __name__ == "__main__":
    fix_all()
