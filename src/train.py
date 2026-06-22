import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from imblearn.over_sampling import SMOTE

# =========================
# LOAD DATA
# =========================

print("Loading dataset...")

df = pd.read_csv(
    "data/processed_jobs.csv",
    low_memory=False
)

# Remove missing values
df = df.dropna(subset=["text", "fraudulent"])

# Convert text to string
df["text"] = df["text"].astype(str)

X = df["text"]
y = df["fraudulent"]

print("Dataset shape:", df.shape)

# =========================
# TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# =========================
# TF-IDF VECTORIZATION
# =========================

print("\nVectorizing text...")

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=2000,
    ngram_range=(1, 1),
    min_df=10
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print("Vectorization completed!")

# =========================
# HANDLE CLASS IMBALANCE
# =========================

print("\nApplying SMOTE...")

smote = SMOTE(random_state=42)

X_train_resampled, y_train_resampled = smote.fit_resample(
    X_train_vec,
    y_train
)

print("SMOTE completed!")
print("Resampled shape:", X_train_resampled.shape)

# =========================
# TRAIN MODEL
# =========================

print("\nTraining model...")

model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

model.fit(X_train_resampled, y_train_resampled)

print("Model training completed!")

# =========================
# PREDICTIONS
# =========================

y_pred = model.predict(X_test_vec)

# =========================
# EVALUATION
# =========================

accuracy = accuracy_score(y_test, y_pred)

print("\n==============================")
print("MODEL PERFORMANCE")
print("==============================")

print(f"Accuracy: {accuracy:.4f}")

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

# =========================
# SAVE MODEL
# =========================

joblib.dump(model, "models/model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")

print("\nModel saved successfully!")
print("Saved: models/model.pkl")
print("Saved: models/vectorizer.pkl")