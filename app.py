from flask import Flask, request, jsonify
import pickle
import numpy as np
import os

app = Flask(__name__)

# ── Load model safely ─────────────────────────────────────────────
MODEL_PATH = "model.pkl"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("model.pkl not found. Make sure it is in the project root.")

with open(MODEL_PATH, "rb") as f:
    data = pickle.load(f)

model = data["model"]
labels = data["labels"]

# ── Routes ───────────────────────────────────────────────────────
@app.route("/")
def home():
    return "Gesture API is running"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        body = request.get_json()

        if not body or "landmarks" not in body:
            return jsonify({"error": "Missing 'landmarks' in request"}), 400

        landmarks = body["landmarks"]

        if len(landmarks) != 42:
            return jsonify({"error": "Expected 42 landmark values"}), 400

        arr = np.array(landmarks).reshape(1, -1)

        probs = model.predict_proba(arr)[0]
        idx = int(np.argmax(probs))

        return jsonify({
            "gesture": labels[idx],
            "confidence": float(probs[idx])
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Run server (Render compatible) ───────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # IMPORTANT for deployment
    app.run(host="0.0.0.0", port=port)