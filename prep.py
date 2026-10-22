from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT_DIR = Path(__file__).resolve().parent
DATA_PATH = ROOT_DIR / "data" / "tourism.csv"
OUTPUT_DIR = ROOT_DIR / "data" / "processed"

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    print("Original dataset shape:", df.shape)

    df = df.drop(columns=["Unnamed: 0"], errors="ignore")
    df = df.drop_duplicates()

    target = "ProdTaken"
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' is missing.")

    if df[target].isna().any():
        raise ValueError("Target column contains missing values.")

    for column in df.columns:
        if column == target or not df[column].isna().any():
            continue

        if pd.api.types.is_numeric_dtype(df[column]):
            df[column] = df[column].fillna(df[column].median())
        else:
            mode_values = df[column].mode()
            if not mode_values.empty:
                df[column] = df[column].fillna(mode_values.iloc[0])
            else:
                df[column] = df[column].fillna("Unknown")

    train_df, test_df = train_test_split(
        df,
        test_size=0.20,
        random_state=42,
        stratify=df[target]
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_path = OUTPUT_DIR / "train.csv"
    test_path = OUTPUT_DIR / "test.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("\nDATA PREPARATION SUCCESSFUL")
    print("Cleaned dataset shape:", df.shape)
    print("Training set shape:", train_df.shape)
    print("Testing set shape:", test_df.shape)
    print(f"Training data saved to: {train_path}")
    print(f"Testing data saved to: {test_path}")

if __name__ == "__main__":
    main()
