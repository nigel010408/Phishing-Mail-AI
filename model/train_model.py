# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn import metrics
import joblib

# 1️⃣ Laad trainingsdata
# dataset.csv bevat 2 kolommen: text,label
# Voorbeeld:
# "Uw account is geblokkeerd, klik hier",phishing
# "Win een iPhone gratis",spam
# "Factuur oktober bijgevoegd",important
# "Lunchmeeting om 12:00",normal
df = pd.read_csv("dataset.csv")

# 2️⃣ Train/test split
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42)

# 3️⃣ Bouw pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="dutch", max_features=5000)),
    ("clf", MultinomialNB())
])

# 4️⃣ Train model
model.fit(X_train, y_train)

# 5️⃣ Test accuratesse
pred = model.predict(X_test)
print("Model Accuracy:", metrics.accuracy_score(y_test, pred))
print(metrics.classification_report(y_test, pred))

# 6️⃣ Bewaar model
joblib.dump(model, "model.pkl")
print("✅ Model opgeslagen als model.pkl")
