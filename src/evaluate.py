from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/processed_jobs.csv")

X = df["text"]
y = df["fraudulent"]

vectorizer = joblib.load("models/vectorizer.pkl")
model = joblib.load("models/model.pkl")

X_vec = vectorizer.transform(X)

_, X_test, _, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42
)

ConfusionMatrixDisplay.from_estimator(
    model, X_test, y_test
)

plt.show()