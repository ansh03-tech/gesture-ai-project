"""
train.py — Gesture Model Training (FIXED + VERBOSE)
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder


def train_model(data_path: str, model_path: str) -> dict:
    # ── Load data ─────────────────────────────────────────────
    if not os.path.exists(data_path):
        print("❌ ERROR: gestures_data.csv not found")
        return {"success": False}

    df = pd.read_csv(data_path, header=None)

    print(f"📊 Loaded {len(df)} samples")

    if df.shape[0] < 10:
        print("❌ ERROR: Need at least 10 samples")
        return {"success": False}

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # Encode labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    labels = list(le.classes_)

    print(f"🧠 Detected gestures: {labels}")

    if len(labels) < 2:
        print("❌ ERROR: Need at least 2 different gestures")
        return {"success": False}

    # ── Train ────────────────────────────────────────────────
    print("🚀 Training model...")

    clf = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    clf.fit(X, y_enc)

    # ── Cross-validation ─────────────────────────────────────
    cv = StratifiedKFold(n_splits=min(5, len(labels)), shuffle=True, random_state=42)
    cv_scores = cross_val_score(clf, X, y_enc, cv=cv, scoring="accuracy")

    cv_mean = float(np.mean(cv_scores))
    cv_std = float(np.std(cv_scores))

    # ── Evaluation ───────────────────────────────────────────
    y_pred = clf.predict(X)
    report = classification_report(y_enc, y_pred, target_names=labels, output_dict=True)
    train_acc = float(report["accuracy"])

    # ── Save (IMPORTANT FIX) ─────────────────────────────────
    model_path = "model.pkl"   # force root save

    with open(model_path, "wb") as f:
        pickle.dump({
            "model": clf,
            "labels": labels,
            "encoder": le
        }, f)

    print("✅ Model saved as model.pkl")
    print(f"🎯 Train Accuracy: {round(train_acc * 100, 2)}%")
    print(f"📈 CV Accuracy: {round(cv_mean * 100, 2)}%")

    return {"success": True}


def load_model(model_path: str):
    if not os.path.exists(model_path):
        return None
    with open(model_path, "rb") as f:
        return pickle.load(f)


# ─────────────────────────────────────────────
# ✅ ENTRY POINT (VERY IMPORTANT)
# ─────────────────────────────────────────────
if __name__ == "__main__":
    DATA_PATH = "gestures_data.csv"

    result = train_model(DATA_PATH, "model.pkl")

    if not result["success"]:
        print("❌ Training failed")