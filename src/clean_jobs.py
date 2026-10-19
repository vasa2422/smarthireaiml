
from pathlib import Path
import pandas as pd

# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = PROJECT_DIR / "data" / "raw" / "postings.csv"
OUTPUT_DIR = PROJECT_DIR / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "cleaned_jobs.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading job postings...")
df = pd.read_csv(INPUT_FILE, low_memory=False)

print("Original number of jobs:", len(df))

# Keep only the columns needed for SmartHire
required_columns = ["title", "description", "company_name", "location"]
missing_columns = [c for c in required_columns if c not in df.columns]

if missing_columns:
    raise ValueError(f"Missing expected columns: {missing_columns}")

jobs = df[required_columns].copy()

# Remove records without a title or description
jobs = jobs.dropna(subset=["title", "description"])

# Remove empty titles and descriptions
jobs["title"] = jobs["title"].astype(str).str.strip()
jobs["description"] = jobs["description"].astype(str).str.strip()

jobs = jobs[
    (jobs["title"] != "") &
    (jobs["description"] != "")
].copy()

# Clean company and location fields
for column in ["company_name", "location"]:
    jobs[column] = jobs[column].fillna("Not specified").astype(str).str.strip()
    jobs.loc[jobs[column] == "", column] = "Not specified"

# Remove duplicate job records
jobs = jobs.drop_duplicates(
    subset=["title", "company_name", "location", "description"]
)

# Save cleaned data
jobs.to_csv(OUTPUT_FILE, index=False)

print("\n===== JOB DATA CLEANING RESULTS =====")
print("Cleaned number of jobs:", len(jobs))
print("Removed records:", len(df) - len(jobs))
print("Saved file:", OUTPUT_FILE)

print("\nSample cleaned jobs:")
print(jobs[["title", "company_name", "location"]].head(10).to_string(index=False))
