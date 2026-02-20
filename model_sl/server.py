from flask import Flask, request, jsonify
import joblib
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
model = joblib.load(MODEL_PATH)

@app.route("/classify", methods=["POST"])
def classify():
    data = request.json
    subject = data.get("subject", "")
    body = data.get("body", "")

    combined = f"Subject: {subject}\nBody: {body}"
    prediction = model.predict([combined])[0]
    proba = model.predict_proba([combined])[0]

    confidence = round(max(proba), 2)
    label = "phishing" if prediction == 1 else "legitimate"

    return jsonify({
        "label": label,
        "confidence": confidence
    })

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000)
