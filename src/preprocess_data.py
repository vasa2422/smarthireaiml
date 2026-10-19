
from pathlib import Path
import pandas as pd
import re

# Find the project directory
PROJECT_DIR = Path(__file__).resolve().parent.parent

# File locations
INPUT_FILE = PROJECT_DIR / "data" / "raw" / "UpdatedResumeDataSet.csv"
OUTPUT_DIR = PROJECT_DIR / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "cleaned_resumes.csv"

# Create output folder if needed
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load dataset
df = pd.read_csv(INPUT_FILE)

print("Original columns:", df.columns.tolist())

# Check required columns
if "Category" not in df.columns or "Resume" not in df.columns:
    raise ValueError(
        "Expected 'Category' and 'Resume' columns. "
        f"Found: {df.columns.tolist()}"
    )

# Remove missing and duplicate records
df = df.dropna(subset=["Category", "Resume"]).copy()
df = df.drop_duplicates(subset=["Category", "Resume"])

# Clean resume text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9+#.\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

df["Resume"] = df["Resume"].apply(clean_text)

# Remove empty resumes
df = df[df["Resume"].str.len() > 0].copy()

# Save processed data
df.to_csv(OUTPUT_FILE, index=False)

print("\nData cleaning completed!")
print("Remaining resumes:", len(df))
print("Number of categories:", df["Category"].nunique())
print("Saved to:", OUTPUT_FILE)
print("\nCategory counts:")
print(df["Category"].value_counts())
