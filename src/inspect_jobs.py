
from pathlib import Path
import pandas as pd

# Project directory
PROJECT_DIR = Path(__file__).resolve().parent.parent
JOBS_FILE = PROJECT_DIR / "data" / "raw" / "postings.csv"

print("Loading job dataset...")

# Read CSV safely
df = pd.read_csv(JOBS_FILE, low_memory=False)

print("\n===== JOB DATASET INSPECTION =====")
print("Number of rows:", len(df))
print("Number of columns:", len(df.columns))

print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst 5 job titles:")
if "title" in df.columns:
    print(df["title"].head())
else:
    print("No 'title' column found.")

print("\nMissing values in important columns:")
for column in ["title", "description", "company_name", "location"]:
    if column in df.columns:
        print(f"{column}: {df[column].isna().sum()}")

print("\nSample job records:")
available_columns = [
    col for col in ["title", "company_name", "location", "description"]
    if col in df.columns
]
print(df[available_columns].head(3).to_string(index=False, max_colwidth=120))
