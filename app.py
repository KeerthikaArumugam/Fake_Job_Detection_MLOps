import streamlit as st
import joblib
import pandas as pd
import os
from datetime import datetime

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

        fake_probability = probability * 100
        genuine_probability = (1 - probability) * 100

        st.markdown("## Prediction Result")

        if prediction == 1:

            st.error("🚨 Fake Job Posting Detected!")

            st.metric(
                label="Fake Job Probability",
                value=f"{fake_probability:.2f}%"
            )

            prediction_label = "Fake Job"

        else:

            st.success("✅ Genuine Job Posting")

            st.metric(
                label="Genuine Job Confidence",
                value=f"{genuine_probability:.2f}%"
            )

            prediction_label = "Genuine Job"

        st.markdown("---")

        st.subheader("Prediction Summary")

        st.write(f"""
- **Prediction:** {prediction_label}
- **Fake Probability:** {fake_probability:.2f}%
- **Genuine Probability:** {genuine_probability:.2f}%
""")

        # =====================================
        # SAVE PREDICTION LOG
        # =====================================

        log_data = {
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Prediction": prediction_label,
            "Fake Probability (%)": round(fake_probability, 2),
            "Genuine Probability (%)": round(genuine_probability, 2),
            "Job Description": job_text
        }

        log_df = pd.DataFrame([log_data])

        if os.path.exists("prediction_logs.csv"):
            log_df.to_csv(
                "prediction_logs.csv",
                mode="a",
                header=False,
                index=False
            )
        else:
            log_df.to_csv(
                "prediction_logs.csv",
                index=False
            )

# =========================
# FOOTER
# =========================

st.markdown("---")

st.caption(
    "Fake Job Detection using NLP, TF-IDF, Logistic Regression, SMOTE and Streamlit"
)