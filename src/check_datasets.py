from pathlib import Path
import pandas as pd

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"

files = [
    DATA_DIR / "cleaned_jobs.csv",
    DATA_DIR / "cleaned_resumes.csv",
]

for file_path in files:
    print("\n" + "=" * 60)
    print(f"DATASET: {file_path.name}")
    print("=" * 60)

    if not file_path.exists():
        print(f"File not found: {file_path}")
        continue

    try:
        df = pd.read_csv(file_path, low_memory=False, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, low_memory=False, encoding="latin-1")

    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    print("\nColumn names:")
    for column in df.columns:
        print(f"- {column}")

    print("\nMissing values:")
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    if missing.empty:
        print("No missing values.")
    else:
        print(missing.to_string())

    print(f"\nDuplicate rows: {df.duplicated().sum():,}")

    print("\nFirst 3 records:")
    print(df.head(3).to_string(index=False, max_cols=8, max_colwidth=35))

print("\nDataset inspection completed.")