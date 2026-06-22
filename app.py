import streamlit as st
import joblib

# =========================
# LOAD MODEL
# =========================

model = joblib.load("models/model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Fake Job Detection",
    page_icon="🔍",
    layout="centered"
)

# =========================
# TITLE
# =========================

st.title("🔍 Fake Job Posting Detection System")

st.write(
    "This application uses Machine Learning and NLP to identify whether a job posting is Genuine or Fake."
)

st.markdown("---")

# =========================
# INPUT
# =========================

job_text = st.text_area(
    "Paste Job Description",
    height=250,
    placeholder="Paste the complete job description here..."
)

# =========================
# PREDICTION
# =========================

if st.button("Predict"):

    if job_text.strip() == "":
        st.warning("⚠ Please enter a job description.")

    else:

        # Vectorize
        text_vector = vectorizer.transform([job_text])

        # Prediction
        prediction = model.predict(text_vector)[0]

        # Probability
        probability = model.predict_proba(text_vector)[0][1]

        st.markdown("## Prediction Result")

        if prediction == 1:

            st.error("🚨 Fake Job Posting Detected!")

            st.metric(
                label="Fake Job Probability",
                value=f"{probability*100:.2f}%"
            )

        else:

            st.success("✅ Genuine Job Posting")

            st.metric(
                label="Genuine Job Confidence",
                value=f"{(1-probability)*100:.2f}%"
            )

        st.markdown("---")

        st.subheader("Prediction Summary")

        st.write(
            f"""
            - Prediction: {'Fake Job' if prediction == 1 else 'Genuine Job'}
            - Fake Probability: {probability*100:.2f}%
            - Genuine Probability: {(1-probability)*100:.2f}%
            """
        )

# =========================
# FOOTER
# =========================

st.markdown("---")

st.caption(
    "Fake Job Detection using NLP, TF-IDF, Logistic Regression, SMOTE and Streamlit"
)