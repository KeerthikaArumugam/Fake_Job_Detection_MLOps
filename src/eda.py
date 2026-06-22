import pandas as pd

df = pd.read_csv("data/fake_job_postings.csv")

print("Missing Values:\n")
print(df.isnull().sum())