"""
GestureScript V1.5 — Main Backend Server
Flask API that serves gesture recognition, training, and profile management.
"""

import os
import cv2
import pickle
import numpy as np
import mediapipe as mp
import base64
import time
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from collections import deque

# Import local modules
from collector import extract_landmarks
from train import train_model, load_model
from actions import execute_action
from profile_manager import load_profile, save_profile, list_profiles

app = Flask(__name__)
CORS(app)  # Allow frontend to communicate with backend

# ── Global State ─────────────────────────────────────────────────────────────
MODEL_PATH = "../model/model.pkl"
DATA_PATH  = "../model/gestures_data.csv"

model_data = load_model(MODEL_PATH)   # {"model": clf, "labels": [...]}
active_profile_name = "default"
active_profile = load_profile(active_profile_name)

# Robustness: hold timer + cooldown
CONFIDENCE_THRESHOLD = 0.85   # 85 %
HOLD_FRAMES_REQUIRED  = 15    # ~0.5 sec at 30 fps
COOLDOWN_SECONDS      = 2.0

gesture_buffer  = deque(maxlen=HOLD_FRAMES_REQUIRED)  # last N predictions
last_action_time = 0.0
last_action_name = "—"

# MediaPipe hands
mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
)

# ── Helper ────────────────────────────────────────────────────────────────────

def decode_frame(b64_string: str) -> np.ndarray:
    """Decode a base64-encoded JPEG frame from the frontend."""
    img_bytes = base64.b64decode(b64_string)
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return frame


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/predict", methods=["POST"])
def predict():
    """
    Receive a webcam frame, run MediaPipe + classifier, return prediction.
    Body JSON: { "frame": "<base64 jpeg>" }
    """
    global last_action_time, last_action_name, model_data

    data = request.get_json()
    if not data or "frame" not in data:
        return jsonify({"error": "No frame provided"}), 400

    frame = decode_frame(data["frame"])
    if frame is None:
        return jsonify({"error": "Could not decode frame"}), 400

    # Extract 21 hand-landmark coordinates (42 features: x,y per landmark)
    landmarks = extract_landmarks(frame, hands_detector)

    if landmarks is None:
        gesture_buffer.clear()
        return jsonify({
            "gesture": None,
            "confidence": 0.0,
            "action": None,
            "profile": active_profile_name,
            "last_action": last_action_name,
            "hand_detected": False,
        })

    # ── Classify ──────────────────────────────────────────────────────────────
    if model_data is None:
        return jsonify({"error": "Model not trained yet"}), 503

    clf    = model_data["model"]
    labels = model_data["labels"]

    probs      = clf.predict_proba([landmarks])[0]
    pred_idx   = int(np.argmax(probs))
    confidence = float(probs[pred_idx])
    gesture    = labels[pred_idx] if confidence >= CONFIDENCE_THRESHOLD else None

    # ── Hold Timer ────────────────────────────────────────────────────────────
    gesture_buffer.append(gesture)

    # Gesture is "stable" only if the buffer is full AND all same gesture
    stable_gesture = None
    if len(gesture_buffer) == HOLD_FRAMES_REQUIRED:
        if all(g == gesture_buffer[0] for g in gesture_buffer) and gesture_buffer[0]:
            stable_gesture = gesture_buffer[0]

    # ── Cooldown + Action ─────────────────────────────────────────────────────
    action_fired = None
    now = time.time()
    if stable_gesture and (now - last_action_time) > COOLDOWN_SECONDS:
        mapped_action = active_profile.get(stable_gesture)
        if mapped_action:
            execute_action(mapped_action)
            action_fired     = mapped_action
            last_action_time = now
            last_action_name = f"{stable_gesture} → {mapped_action}"

    return jsonify({
        "gesture":      stable_gesture or gesture,
        "confidence":   round(confidence * 100, 1),
        "stable":       stable_gesture is not None,
        "action":       action_fired,
        "profile":      active_profile_name,
        "last_action":  last_action_name,
        "hand_detected": True,
    })


@app.route("/collect", methods=["POST"])
def collect():
    """
    Save a frame's landmark data labelled with a gesture name.
    Body JSON: { "frame": "<base64>", "label": "thumbs_up" }
    """
    data = request.get_json()
    frame = decode_frame(data["frame"])
    label = data.get("label", "unknown")

    landmarks = extract_landmarks(frame, hands_detector)
    if landmarks is None:
        return jsonify({"saved": False, "reason": "No hand detected"})

    # Append row to CSV
    row = ",".join(map(str, landmarks)) + f",{label}\n"
    os.makedirs("../model", exist_ok=True)
    with open(DATA_PATH, "a") as f:
        f.write(row)

    return jsonify({"saved": True, "label": label})


@app.route("/train", methods=["POST"])
def train():
    """Retrain the model from the current CSV data."""
    global model_data
    result = train_model(DATA_PATH, MODEL_PATH)
    if result["success"]:
        model_data = load_model(MODEL_PATH)
    return jsonify(result)


@app.route("/gestures", methods=["GET"])
def list_gestures():
    """Return unique gesture labels present in training data."""
    if not os.path.exists(DATA_PATH):
        return jsonify({"gestures": []})
    gestures = set()
    with open(DATA_PATH) as f:
        for line in f:
            parts = line.strip().split(",")
            if parts:
                gestures.add(parts[-1])
    return jsonify({"gestures": sorted(gestures)})


@app.route("/profiles", methods=["GET"])
def get_profiles():
    return jsonify({"profiles": list_profiles()})


@app.route("/profile/<name>", methods=["GET"])
def get_profile(name):
    return jsonify(load_profile(name))


@app.route("/profile/<name>", methods=["POST"])
def update_profile(name):
    global active_profile, active_profile_name
    mapping = request.get_json()
    save_profile(name, mapping)
    if name == active_profile_name:
        active_profile = mapping
    return jsonify({"saved": True})


@app.route("/switch_profile/<name>", methods=["POST"])
def switch_profile(name):
    global active_profile, active_profile_name
    active_profile      = load_profile(name)
    active_profile_name = name
    return jsonify({"active": name, "mapping": active_profile})


@app.route("/status", methods=["GET"])
def status():
    trained = model_data is not None
    gestures = model_data["labels"] if trained else []
    return jsonify({
        "model_trained": trained,
        "gestures": gestures,
        "active_profile": active_profile_name,
        "profile_mapping": active_profile,
    })


if __name__ == "__main__":
    print("🚀 GestureScript V1.5 backend running at http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
