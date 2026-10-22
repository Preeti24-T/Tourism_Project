from pathlib import Path
import pandas as pd
import sys

# Locate the dataset inside the repository relative to current working directory
ROOT_DIR = Path.cwd() / "tourism_project"
DATA_PATH = ROOT_DIR / "data" / "tourism.csv"

# Expected dataset columns
EXPECTED_COLUMNS = [
    "CustomerID", "ProdTaken", "Age", "TypeofContact",
    "CityTier", "DurationOfPitch", "Occupation", "Gender",
    "NumberOfPersonVisiting", "NumberOfFollowups",
    "ProductPitched", "PreferredPropertyStar", "MaritalStatus",
    "NumberOfTrips", "Passport", "PitchSatisfactionScore",
    "OwnCar", "NumberOfChildrenVisiting", "Designation",
    "MonthlyIncome"
]

def main():
    if not DATA_PATH.exists():
        sys.exit(f"ERROR: Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    missing_columns = set(EXPECTED_COLUMNS) - set(df.columns)

    if missing_columns:
        sys.exit(
            f"ERROR: Missing expected columns: {sorted(missing_columns)}"
        )

    if df.empty:
        sys.exit("ERROR: Dataset is empty.")

    print("DATA REGISTRATION SUCCESSFUL")
    print(f"Dataset path: {DATA_PATH}")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns in CSV: {df.shape[1]}")
    print(f"Missing values: {df.isna().sum().sum()}")
    print(f"Duplicate rows: {df.duplicated().sum()}")
    print("All expected columns are present.")

if __name__ == "__main__":
    main()
