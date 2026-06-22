import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load data
df = pd.read_csv("data/processed_jobs.csv")

# Features and target
X = df["text"]
y = df["fraudulent"]

# Split FIRST (important)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=8000,
    ngram_range=(1, 2)
)

# Transform text data
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Model (balanced for class imbalance)
model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

# Train model
model.fit(X_train_vec, y_train)

# Predictions
y_pred = model.predict(X_test_vec)

# =========================
# MODEL EVALUATION
# =========================

print("\n===== MODEL PERFORMANCE =====")

print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# =========================
# SAVE MODEL
# =========================

joblib.dump(model, "models/model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")

print("\nModel retrained and saved successfully!")