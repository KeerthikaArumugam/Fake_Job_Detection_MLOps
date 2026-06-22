import pandas as pd

# Load dataset
df = pd.read_csv("data/fake_job_postings.csv")

# Fill missing values
df.fillna('', inplace=True)

# Create a single text column
df['text'] = (
    df['title'] + ' ' +
    df['company_profile'] + ' ' +
    df['description'] + ' ' +
    df['requirements'] + ' ' +
    df['benefits']
)

# Keep only text and target
df = df[['text', 'fraudulent']]

print(df.head())
print("\nShape:", df.shape)

# Save processed dataset
df.to_csv("data/processed_jobs.csv", index=False)

print("\nProcessed dataset saved successfully!")