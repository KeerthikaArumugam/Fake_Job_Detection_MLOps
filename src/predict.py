import joblib

# Load model
model = joblib.load("models/model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

# Sample job posting
job_text = """
Urgently hiring Data Entry Operators.
Work from home.
No experience needed.
Earn $5000 per week.
"""

# Transform text
job_vector = vectorizer.transform([job_text])

# Predict
prediction = model.predict(job_vector)

if prediction[0] == 1:
    print("Fraudulent Job Posting")
else:
    print("Genuine Job Posting")