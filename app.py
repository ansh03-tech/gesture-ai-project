from flask import Flask, request, jsonify, send_from_directory
import pickle
import numpy as np
import os

app = Flask(__name__, static_folder=".", static_url_path="")

# Load model
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

# ── API ────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json["landmarks"]
        arr = np.array(data).reshape(1, -1)

        probs = model.predict_proba(arr)[0]
        idx = int(np.argmax(probs))

        return jsonify({
            "gesture": labels[idx],
            "confidence": float(probs[idx])
        })

    except Exception as e:
        return jsonify({"error": str(e)})

# ── Run ───────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)