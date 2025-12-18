import joblib
import os

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

model = joblib.load(MODEL_PATH)

def classify_email(subject, text):
    combined = f"{subject} {text}"
    prediction = model.predict([combined])[0]
    proba = model.predict_proba([combined])[0]
    confidence = round(max(proba), 2)
    label = "phishing" if prediction == 1 else "legitimate"
    return label, confidence

if __name__ == "__main__":
    # Voorbeeld
    subj = input("subject: ")
    body = input("body: ")
    label, conf = classify_email(subj, body)
    print(f"🔍 Resultaat: {label} (zekerheid {conf})")
