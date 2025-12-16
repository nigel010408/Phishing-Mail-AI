import os
import pandas as pd
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn import metrics

# Pad naar dataset
BASE_DIR = os.path.dirname(__file__)
DATASET_PATH = os.path.join(BASE_DIR, "../datasets/phishing_legit_dataset_cleaned_v2.xlsx")
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")
LOG_PATH = os.path.join(BASE_DIR, "training_log.csv")

def load_dataset():
    print(f"Laden van dataset: {DATASET_PATH}")
    df = pd.read_excel(DATASET_PATH)

    # Controleer kolommen
    required = {"Subject", "Text", "label"}
    if not required.issubset(df.columns):
        raise ValueError(f"Dataset mist vereiste kolommen: {required}")

    # Combineer subject + tekst in één veld
    df["combined_text"] = df["Subject"].astype(str) + " " + df["Text"].astype(str)

    # Zorg dat labels int zijn
    df["label"] = df["label"].astype(int)

    print(f"Dataset geladen ({len(df)} rijen).")
    return df[["combined_text", "label"]]

def train_model():
    df = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        df["combined_text"], df["label"], test_size=0.2, random_state=42
    )

    # Maak pipeline
    model = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", max_features=5000)),
        ("clf", MultinomialNB())
    ])

    print("Model wordt getraind...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = metrics.accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.3f}")
    print(metrics.classification_report(y_test, y_pred))

    # Sla op
    joblib.dump(model, MODEL_PATH)
    print(f"Model opgeslagen als {MODEL_PATH}")

    # Vectorizer apart opslaan voor scan_emails.py
    vectorizer = model.named_steps["tfidf"]
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print(f"Vectorizer opgeslagen als {VECTORIZER_PATH}")

    # Log training
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_PATH, "a") as f:
        f.write(f"{timestamp},{acc:.3f}\n")
    print(f"Log bijgewerkt ({LOG_PATH})")

if __name__ == "__main__":
    train_model()
