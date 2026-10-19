
from pathlib import Path
import pandas as pd

# Locate the project folder and dataset
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_DIR / "data" / "raw" / "UpdatedResumeDataSet.csv"

# Check whether the dataset exists
if not DATASET_PATH.exists():
    print("Dataset not found!")
    print("Expected location:", DATASET_PATH)
else:
    # Load the dataset
    df = pd.read_csv(DATASET_PATH)

    print("\n===== SMART HIRE: RESUME DATASET =====")
    print("Dataset loaded successfully!")
    print("Number of rows:", df.shape[0])
    print("Number of columns:", df.shape[1])

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nResume categories:")
    if "Category" in df.columns:
        print(df["Category"].value_counts())
    else:
        print("Category column not found. Check the column names above.")
