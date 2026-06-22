# Fake Job Posting Detection using NLP and MLOps

## Project Overview
This project detects whether a job posting is genuine or fraudulent using Natural Language Processing and Machine Learning.

## Technologies Used
- Python
- Pandas
- Scikit-Learn
- TF-IDF
- Random Forest
- FastAPI
- MLflow

## Dataset
Fake Job Postings Dataset (Kaggle)

## Model Performance
Accuracy: 98.10%
Precision: 100%
Recall: 62.43%
F1 Score: 76.87%

## Run Project

Train Model:
python src/train.py

Predict:
python src/predict.py

API:
uvicorn app.main:app --reload