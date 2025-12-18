import os
import pandas as pd
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn import metrics

# Dataset paths
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")
LOG_PATH = os.path.join(BASE_DIR, "training_log.csv")

def load_dataset():
    print("Loading Hugging Face phishing dataset...")

    splits = {
        "train": "data/train-00000-of-00001.parquet",
        "test": "data/test-00000-of-00001.parquet"
    }

    df_train = pd.read_parquet(
        "hf://datasets/drorrabin/phishing_emails-data/" + splits["train"]
    )
    df_test = pd.read_parquet(
        "hf://datasets/drorrabin/phishing_emails-data/" + splits["test"]
    )

    print(f"Training rows: {len(df_train)}, Test rows: {len(df_test)}")

    # Expected columns: text, label
    required = {"text", "email_type"}
    print(df_train.columns)
    if not required.issubset(df_train.columns):
        raise ValueError(f"Dataset is missing required columns: {required}")

    # Ensure correct types
    label_map = {
        "safe email": 0,
        "phishing email": 1
    }

    df_train["text"] = df_train["text"].astype(str)
    df_test["text"] = df_test["text"].astype(str)
    df_train["label"] = df_train["email_type"].map(label_map)
    df_test["label"] = df_test["email_type"].map(label_map)

    return df_train, df_test

def train_model():
    df_train, df_test = load_dataset()

    X_train = df_train["text"]
    y_train = df_train["label"]
    X_test = df_test["text"]
    y_test = df_test["label"]

    # Create pipeline
    model = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", max_features=5000)),
        ("clf", MultinomialNB())
    ])

    print("Training model...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = metrics.accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.3f}")
    print(metrics.classification_report(y_test, y_pred))

    # Save model
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved as {MODEL_PATH}")

    # Save vectorizer separately for scan_emails.py
    vectorizer = model.named_steps["tfidf"]
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print(f"Vectorizer saved as {VECTORIZER_PATH}")

    # Log training
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_PATH, "a") as f:
        f.write(f"{timestamp},{acc:.3f}\n")
    print(f"Training log updated ({LOG_PATH})")

if __name__ == "__main__":
    train_model()
