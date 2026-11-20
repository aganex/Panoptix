
from pathlib import Path
import json

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib


# ==========================================
# CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "dataset" / "raw" / "hardware_metrics.csv"
OUTPUT_DIR = BASE_DIR / "dataset" / "processed"

FEATURES = [
    "cpu_usage",
    "ram_usage",
]

TARGET = "load_category"

LABEL_MAPPING = {
    "Low": 0,
    "Moderate": 1,
    "High": 2,
}

TEST_SIZE = 0.20
RANDOM_STATE = 42


# ==========================================
# DATA VALIDATION
# ==========================================

def load_dataset():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Dataset tidak ditemukan: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    required_columns = FEATURES + [TARGET]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Kolom tidak ditemukan: {missing_columns}"
        )

    if df.empty:
        raise ValueError("Dataset kosong.")

    return df


# ==========================================
# DATA CLEANING
# ==========================================

def clean_dataset(df):
    data = df[FEATURES + [TARGET]].copy()

    # Convert features to numeric
    for column in FEATURES:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )

    # Remove invalid feature values
    data = data.dropna(subset=FEATURES + [TARGET])

    # Validate percentage values
    for column in FEATURES:
        data = data[
            data[column].between(0, 100)
        ]

    # Validate labels
    data = data[
        data[TARGET].isin(LABEL_MAPPING)
    ]

    if data.empty:
        raise ValueError(
            "Tidak ada data valid setelah cleaning."
        )

    return data.reset_index(drop=True)


# ==========================================
# PREPROCESSING
# ==========================================

def preprocess():
    print("=" * 50)
    print("PANOPTIX - DATA PREPROCESSING")
    print("=" * 50)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Load
    df = load_dataset()

    print(f"\nTotal data awal : {len(df)}")

    # Clean
    data = clean_dataset(df)

    print(f"Total data valid: {len(data)}")

    # Features and target
    X = data[FEATURES]
    y = data[TARGET].map(LABEL_MAPPING)

    # Verify all classes exist
    if set(y.unique()) != set(LABEL_MAPPING.values()):
        raise ValueError(
            "Dataset harus memiliki ketiga kategori: "
            "Low, Moderate, dan High."
        )

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # Fit scaler only on training data
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert back to DataFrame
    train_df = pd.DataFrame(
        X_train_scaled,
        columns=FEATURES
    )

    test_df = pd.DataFrame(
        X_test_scaled,
        columns=FEATURES
    )

    train_df["label"] = y_train.reset_index(drop=True)
    test_df["label"] = y_test.reset_index(drop=True)

    # Save processed datasets
    train_df.to_csv(
        OUTPUT_DIR / "train.csv",
        index=False
    )

    test_df.to_csv(
        OUTPUT_DIR / "test.csv",
        index=False
    )

    # Save scaler
    joblib.dump(
        scaler,
        OUTPUT_DIR / "scaler.pkl"
    )

    # Save metadata
    metadata = {
        "features": FEATURES,
        "target": TARGET,
        "label_mapping": LABEL_MAPPING,
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "scaling": "StandardScaler",
        "total_samples": len(data),
        "training_samples": len(train_df),
        "testing_samples": len(test_df),
    }

    with open(
        OUTPUT_DIR / "metadata.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            metadata,
            file,
            indent=4
        )

    # Display results
    print("\n--- DATA SPLITTING ---")
    print(f"Training: {len(train_df)}")
    print(f"Testing : {len(test_df)}")

    print("\n--- TRAINING DISTRIBUTION ---")
    print(
        y_train.map({
            0: "Low",
            1: "Moderate",
            2: "High"
        }).value_counts()
    )

    print("\n--- TESTING DISTRIBUTION ---")
    print(
        y_test.map({
            0: "Low",
            1: "Moderate",
            2: "High"
        }).value_counts()
    )

    print("\n--- OUTPUT FILES ---")
    print(OUTPUT_DIR / "train.csv")
    print(OUTPUT_DIR / "test.csv")
    print(OUTPUT_DIR / "scaler.pkl")
    print(OUTPUT_DIR / "metadata.json")

    print("\nPreprocessing selesai.")


if __name__ == "__main__":
    preprocess()
