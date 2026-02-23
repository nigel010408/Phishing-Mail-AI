import os
import pandas as pd
import joblib
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn import metrics

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")
LOG_PATH = os.path.join(BASE_DIR, "training_log.csv")

# data preprocessing

def extract_subject_body(text):
    """
    Extract subject and body from dataset text
    and return clean formatted string.
    """
    subject = ""
    body = text

    if "Email Subject:" in text:
        try:
            after_subject = text.split("Email Subject:", 1)[1]

            if "Email Body:" in after_subject:
                subject, body = after_subject.split("Email Body:", 1)
            else:
                subject = after_subject
                body = ""

        except Exception:
            pass

    return f"Subject: {subject.strip()}\nBody: {body.strip()}"

# dataset loading

def load_dataset():
    print("Laden van Hugging Face phishing dataset...")

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

    print(f"Train rows: {len(df_train)}, Test rows: {len(df_test)}")

    required = {"text", "email_type"}
    if not required.issubset(df_train.columns):
        raise ValueError(f"Dataset mist vereiste kolommen: {required}")

    label_map = {
        "phishing email": 1,
        "safe email": 0
    }

    df_train["text"] = df_train["text"].astype(str)
    df_test["text"] = df_test["text"].astype(str)
    df_train["label"] = df_train["email_type"].map(label_map).astype(int)
    df_test["label"] = df_test["email_type"].map(label_map).astype(int)

    df_train["clean_text"] = df_train["text"].apply(extract_subject_body)
    df_test["clean_text"] = df_test["text"].apply(extract_subject_body)

    return df_train, df_test

# model training

def train_model():
    df_train, df_test = load_dataset()

    X_train = df_train["clean_text"]
    y_train = df_train["label"]
    X_test = df_test["clean_text"]
    y_test = df_test["label"]

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            stop_words="english",
            max_features=5000,
            ngram_range=(1, 2)
        )),
        ("clf", MultinomialNB())
    ])

    print("Model wordt getraind...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = metrics.accuracy_score(y_test, y_pred)

    print(f"Accuracy: {acc:.3f}")
    print(metrics.classification_report(y_test, y_pred))

    joblib.dump(model, MODEL_PATH)
    print(f"Model opgeslagen als {MODEL_PATH}")

    vectorizer = model.named_steps["tfidf"]
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print(f"Vectorizer opgeslagen als {VECTORIZER_PATH}")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_PATH, "a") as f:
        f.write(f"{timestamp},{acc:.3f}\n")

    print(f"Log bijgewerkt ({LOG_PATH})")


if __name__ == "__main__":
    train_model()
