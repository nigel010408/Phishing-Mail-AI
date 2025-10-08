import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib

# 1. Data inladen
df = pd.read_csv("datasets/CEAS_08.csv")

# 2. NaN-waarden verwijderen
df = df.dropna(subset=["sender", "receiver", "date", "subject", "body", "label", "urls"])

# 3. Labels omzetten naar 0 en 1
# df["label"] = df["label"].map({"phishing": 1, "safe": 0})

# 4. Alle tekstkenmerken combineren (zet alles naar string)
df["combined_text"] = (
    df["sender"].astype(str) + " " +
    df["receiver"].astype(str) + " " +
    df["date"].astype(str) + " " +
    df["subject"].astype(str) + " " +
    df["body"].astype(str) + " " +
    df["urls"].astype(str)
)

# 5. Tekst vectoriseren
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["combined_text"])
y = df["label"]

# 6. Data splitsen
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Model trainen
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# 8. Model testen
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Modelnauwkeurigheid: {accuracy:.2f}")

# 9. Model en vectorizer opslaan
joblib.dump(model, "model/phishing_model.pkl")
joblib.dump(vectorizer, "model/vectorizer.pkl")

print("✅ Model en vectorizer opgeslagen!")
