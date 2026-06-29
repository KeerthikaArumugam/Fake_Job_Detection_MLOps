import os
import joblib
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from imblearn.over_sampling import SMOTE

# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "processed_jobs.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "models", "vectorizer.pkl")

# ============================================================
# MLFLOW CONFIGURATION
# ============================================================

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Fake Job Detection")

# ============================================================
# START MLFLOW RUN
# ============================================================

with mlflow.start_run():

    print("Loading dataset...")

    df = pd.read_csv(DATA_PATH, low_memory=False)

    df = df.dropna(subset=["text", "fraudulent"])

    df["text"] = df["text"].astype(str)

    X = df["text"]
    y = df["fraudulent"]

    print("Dataset shape:", df.shape)

    # ========================================================
    # SPLIT
    # ========================================================

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))

    # ========================================================
    # TF-IDF
    # ========================================================

    print("\nVectorizing text...")

    vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=2000,
    ngram_range=(1,1),
    min_df=20,
    max_df=0.90,
    sublinear_tf=True,
    dtype=np.float32
)

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print("Vectorization completed!")

    # ========================================================
    # SMOTE
    # ========================================================

    print("\nApplying SMOTE...")

    smote = SMOTE(random_state=42)

    X_train_resampled, y_train_resampled = smote.fit_resample(
        X_train_vec,
        y_train
    )

    print("SMOTE completed!")
    print("Resampled shape:", X_train_resampled.shape)

    # ========================================================
    # MODEL
    # ========================================================

    print("\nTraining model...")

    model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    model.fit(X_train_resampled, y_train_resampled)

    print("Model training completed!")

    # ========================================================
    # PREDICTION
    # ========================================================

    y_pred = model.predict(X_test_vec)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("\n==============================")
    print("MODEL PERFORMANCE")
    print("==============================")

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    print("\nClassification Report\n")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix\n")
    print(confusion_matrix(y_test, y_pred))

    # ========================================================
    # SAVE MODEL
    # ========================================================

    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    print("\nModel saved successfully!")

    # ========================================================
    # LOG PARAMETERS
    # ========================================================

    mlflow.log_param("algorithm", "Logistic Regression")
    mlflow.log_param("max_features", 5000)
    mlflow.log_param("ngram_range", "(1,2)")
    mlflow.log_param("min_df", 10)
    mlflow.log_param("smote", True)
    mlflow.log_param("random_state", 42)

    # ========================================================
    # LOG METRICS
    # ========================================================

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    # ========================================================
    # LOG MODEL
    # ========================================================

    mlflow.sklearn.log_model(
        sk_model=model,
        name="FakeJobDetectionModel"
    )

    print("\nModel logged to MLflow successfully!")

print("\nTraining completed successfully!")