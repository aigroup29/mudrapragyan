"""
Step 1 — Extract landmarks from training images
================================================
Reads every image in the dataset, runs MediaPipe Hands,
and saves the 21 normalised hand landmarks to landmarks.csv.

Run:  python scripts/1_extract_landmarks.py
Time: ~20-40 minutes for 87,000 images
"""

import cv2
import numpy as np
import csv
import time
import urllib.request
from pathlib import Path

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

DATASET_DIR = Path(r"C:\Users\DELL\Desktop\MUDRAPRAGYAN.AI PROJECT\dataset\asl_alphabet_train\asl_alphabet_train")
OUTPUT_CSV  = Path(r"C:\Users\DELL\Desktop\MUDRAPRAGYAN.AI PROJECT\landmarks.csv")
MODEL_PATH  = Path(r"C:\Users\DELL\Desktop\MUDRAPRAGYAN.AI PROJECT\scripts\hand_landmarker.task")
MODEL_URL   = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

# Only keep A-Z and space (skip 'del' and 'nothing')
INCLUDE = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ') | {'space'}

# ── Download model if needed ───────────────────────────────────────────────────
if not MODEL_PATH.exists():
    print("Downloading hand landmarker model (~4 MB) ...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Downloaded!\n")

# ── Set up detector ───────────────────────────────────────────────────────────
base_options = mp_python.BaseOptions(model_asset_path=str(MODEL_PATH))
options      = mp_vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.3,
    min_hand_presence_confidence=0.3,
)
detector = mp_vision.HandLandmarker.create_from_options(options)


def process(img_path):
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    img = cv2.resize(img, (224, 224))
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result  = detector.detect(mp_img)
    if not result.hand_landmarks:
        return None
    lm  = result.hand_landmarks[0]
    pts = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)
    sc  = max(float(np.linalg.norm(pts[9] - pts[0])), 1e-4)
    pts = (pts - pts[0]) / sc
    return pts.flatten()


# ── Extract ───────────────────────────────────────────────────────────────────
header = [f'{ax}{i}' for i in range(21) for ax in ['x', 'y', 'z']] + ['label']

total, skipped = 0, 0
start = time.time()

print("Extracting landmarks — this will take a while, go grab a snack!\n")

with open(OUTPUT_CSV, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)

    for class_dir in sorted(DATASET_DIR.iterdir()):
        label = class_dir.name
        if label not in INCLUDE:
            print(f"  Skipping: {label}")
            continue

        images = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
        count  = 0

        for img_path in images:
            vec = process(img_path)
            if vec is None:
                skipped += 1
                continue
            writer.writerow(list(vec) + [label])
            total += 1
            count += 1

        elapsed = time.time() - start
        print(f"  {label:6s}  {count:4d} samples  |  total so far: {total}  |  {elapsed:.0f}s elapsed")

print(f"\nDone!")
print(f"  Saved:   {total} rows  ->  {OUTPUT_CSV}")
print(f"  Skipped: {skipped} images (no hand detected)")
print(f"  Time:    {(time.time()-start)/60:.1f} minutes")
print(f"\nNow run:  python scripts/2_train_model.py")
