import pandas as pd

df = pd.read_csv("data/processed_jobs.csv")

print("Class distribution:")
print(df["fraudulent"].value_counts())

print("\nUnique labels:")
print(df["fraudulent"].unique())