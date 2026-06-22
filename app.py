import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("models/model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

# Page Configuration
st.set_page_config(
    page_title="Fake Job Detection",
    page_icon="🔍",
    layout="centered"
)

# Title
st.title("🔍 Fake Job Posting Detection System")
st.write("Enter a job description below and check whether it is Genuine or Fake.")

# Input Box
job_text = st.text_area(
    "Job Description",
    height=200,
    placeholder="Paste the complete job description here..."
)

# Predict Button
if st.button("Predict"):

    if job_text.strip() == "":
        st.warning("⚠ Please enter a job description.")
    else:
        # Transform text
        text_vector = vectorizer.transform([job_text])

        # Predict
        prediction = model.predict(text_vector)[0]

        # Display Result
        if prediction == 1:
            st.error("🚨 Fake Job Posting Detected!")
        else:
            st.success("✅ Genuine Job Posting")

# Footer
st.markdown("---")
st.caption("Fake Job Detection using Machine Learning (Random Forest + TF-IDF)")