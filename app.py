from flask import Flask, request, jsonify, send_from_directory
import pickle
import numpy as np
import os

app = Flask(__name__, static_folder=".", static_url_path="")

# ── Load model ─────────────────────────────────
with open("model.pkl", "rb") as f:
    data = pickle.load(f)

model = data["model"]
labels = data["labels"]

# ── Serve frontend ─────────────────────────────
@app.route("/")
def serve_index():
    return send_from_directory(".", "index.html")

@app.route("/<path:path>")
def serve_static(path):
    return send_from_directory(".", path)

# ── STATUS (IMPORTANT FIX) ─────────────────────
@app.route("/status")
def status():
    return jsonify({
        "status": "ok",
        "model_trained": True,
        "gestures": labels,
        "active_profile": "default",
        "profile_mapping": {}
    })

# ── PREDICT ────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    try:
        req = request.json

        # Your frontend sends "frame", but model expects landmarks
        # So for now we return a dummy response if frame is sent
        if "frame" in req:
            return jsonify({
                "gesture": "demo",
                "confidence": 50,
                "hand_detected": True,
                "profile": "default",
                "last_action": "none",
                "action": None,
                "stable": False
            })

        # If landmarks are sent (like your earlier test)
        if "landmarks" in req:
            arr = np.array(req["landmarks"]).reshape(1, -1)

            probs = model.predict_proba(arr)[0]
            idx = int(np.argmax(probs))

            return jsonify({
                "gesture": labels[idx],
                "confidence": float(probs[idx] * 100),
                "hand_detected": True,
                "profile": "default",
                "last_action": "none",
                "action": None,
                "stable": True
            })

        return jsonify({"error": "Invalid input format"})

    except Exception as e:
        return jsonify({"error": str(e)})

# ── RUN ───────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)