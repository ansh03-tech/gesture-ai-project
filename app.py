from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load model
with open("model.pkl", "rb") as f:
    data = pickle.load(f)

model = data["model"]
labels = data["labels"]

@app.route("/")
def home():
    return "Gesture API is running"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json["landmarks"]  # list of 42 values
        arr = np.array(data).reshape(1, -1)

        probs = model.predict_proba(arr)[0]
        idx = int(np.argmax(probs))

        return jsonify({
            "gesture": labels[idx],
            "confidence": float(probs[idx])
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)