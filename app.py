from flask import Flask, request, jsonify, send_from_directory
import pickle
import numpy as np
import os
import cv2
import mediapipe as mp
import base64

app = Flask(__name__, static_folder=".", static_url_path="")

# ── Load model ─────────────────────────────────
with open("model.pkl", "rb") as f:
    data = pickle.load(f)

model = data["model"]
labels = data["labels"]

# ── MediaPipe Setup ────────────────────────────
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)

def extract_landmarks(image):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if not result.multi_hand_landmarks:
        return None

    landmarks = []
    for lm in result.multi_hand_landmarks[0].landmark:
        landmarks.extend([lm.x, lm.y])

    return landmarks


# ── PROFILE SYSTEM (FIX) ───────────────────────
PROFILES = {
    "default": {
        "thumbs_up": "volume_up",
        "thumbs_down": "volume_down"
    }
}

active_profile = "default"


# ── Serve frontend ─────────────────────────────
@app.route("/")
def serve_index():
    return send_from_directory(".", "index.html")

@app.route("/<path:path>")
def serve_static(path):
    return send_from_directory(".", path)


# ── STATUS (FIXED) ─────────────────────────────
@app.route("/status")
def status():
    return jsonify({
        "status": "ok",
        "model_trained": True,
        "gestures": labels,
        "active_profile": active_profile,
        "profile_mapping": PROFILES.get(active_profile, {})
    })


# ── PROFILES API (FIXES YOUR ERROR) ────────────
@app.route("/profiles")
def get_profiles():
    return jsonify({
        "profiles": list(PROFILES.keys())
    })

@app.route("/profile/<name>")
def get_profile(name):
    return jsonify(PROFILES.get(name, {}))

@app.route("/profile/<name>", methods=["POST"])
def save_profile(name):
    PROFILES[name] = request.json
    return jsonify({"status": "saved"})

@app.route("/switch_profile/<name>", methods=["POST"])
def switch_profile(name):
    global active_profile
    if name in PROFILES:
        active_profile = name

    return jsonify({
        "mapping": PROFILES.get(active_profile, {})
    })


# ── PREDICT (REAL PIPELINE) ────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    try:
        req = request.json

        if "frame" not in req:
            return jsonify({"error": "No frame provided"})

        # Decode base64 frame
        img_bytes = base64.b64decode(req["frame"])
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Extract landmarks
        landmarks = extract_landmarks(img)

        if landmarks is None:
            return jsonify({
                "hand_detected": False,
                "gesture": None,
                "confidence": 0,
                "profile": active_profile,
                "last_action": None,
                "action": None,
                "stable": False
            })

        arr = np.array(landmarks).reshape(1, -1)

        probs = model.predict_proba(arr)[0]
        idx = int(np.argmax(probs))

        gesture = labels[idx]
        confidence = round(float(probs[idx]) * 100, 2)

        # Get mapped action
        mapping = PROFILES.get(active_profile, {})
        action = mapping.get(gesture, None)

        return jsonify({
            "hand_detected": True,
            "gesture": gesture,
            "confidence": confidence,
            "profile": active_profile,
            "last_action": action,
            "action": action,
            "stable": True
        })

    except Exception as e:
        return jsonify({"error": str(e)})


# ── RUN ───────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)