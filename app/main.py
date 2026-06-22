from fastapi import FastAPI
import joblib

app = FastAPI(
    title="Fake Job Posting Detection API"
)

# Load model and vectorizer
model = joblib.load("models/model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

@app.get("/")
def home():
    return {
        "message": "Fake Job Posting Detection API"
    }

@app.post("/predict")
def predict(job_text: str):

    vector = vectorizer.transform([job_text])

    prediction = model.predict(vector)

    if prediction[0] == 1:
        result = "Fraudulent Job Posting"
    else:
        result = "Genuine Job Posting"

    return {
        "prediction": result
    }