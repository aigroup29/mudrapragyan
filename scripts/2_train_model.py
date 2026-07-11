"""
Step 2 — Train model and export weights as JSON
================================================
Reads landmarks.csv, trains a neural network with sklearn,
and exports weights to model/model_weights.json for the browser.

Run:  python scripts/2_train_model.py
Time: ~5-10 minutes
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

CSV_PATH    = Path(r"C:\Users\DELL\Desktop\MUDRAPRAGYAN.AI PROJECT\landmarks.csv")
MODEL_OUT   = Path(r"C:\Users\DELL\Desktop\MUDRAPRAGYAN.AI PROJECT\model\model_weights.json")
MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)

print("Loading landmarks.csv ...")
df = pd.read_csv(CSV_PATH)
print(f"  {len(df)} rows, {df['label'].nunique()} classes")

X = df.drop('label', axis=1).values.astype(np.float32)
y = df['label'].values

le = LabelEncoder()
y_enc = le.fit_transform(y)

X_train, X_val, y_train, y_val = train_test_split(
    X, y_enc, test_size=0.1, random_state=42, stratify=y_enc
)
print(f"  Train: {len(X_train)}  Val: {len(X_val)}\n")

# ── Scale ──────────────────────────────────────────────────────────────────────
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)

# ── Train ──────────────────────────────────────────────────────────────────────
print("Training neural network ...")
clf = MLPClassifier(
    hidden_layer_sizes=(256, 256, 128),
    activation='relu',
    max_iter=200,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=15,
    random_state=42,
    verbose=True,
)
clf.fit(X_train, y_train)

val_acc = clf.score(X_val, y_val)
print(f"\nValidation accuracy: {val_acc*100:.1f}%")

# ── Export weights as JSON for browser ────────────────────────────────────────
print("\nExporting model weights ...")
export = {
    "labels":     list(le.classes_),
    "scaler_mean": scaler.mean_.tolist(),
    "scaler_std":  scaler.scale_.tolist(),
    "coefs":       [w.tolist() for w in clf.coefs_],
    "intercepts":  [b.tolist() for b in clf.intercepts_],
    "activation":  "relu",
}

with open(MODEL_OUT, 'w') as f:
    json.dump(export, f)

print(f"Model saved to: {MODEL_OUT}")
print(f"Labels: {export['labels']}")
print(f"\nAll done! Validation accuracy: {val_acc*100:.1f}%")
print("Your model is ready for the browser.")
