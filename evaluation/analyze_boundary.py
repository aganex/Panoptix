
from pathlib import Path

import joblib
import pandas as pd
import numpy as np


BASE_DIR = Path(__file__).resolve().parent.parent

PROCESSED_DIR = BASE_DIR / "dataset" / "processed"
EVALUATION_DIR = BASE_DIR / "evaluation"

TRAIN_FILE = PROCESSED_DIR / "train.csv"
SCALER_FILE = PROCESSED_DIR / "scaler.pkl"

BOUNDARY_FILE = EVALUATION_DIR / "consistency_boundary_results.csv"

FEATURES = ["cpu_usage", "ram_usage"]


def main():

    train_df = pd.read_csv(TRAIN_FILE)
    boundary_df = pd.read_csv(BOUNDARY_FILE)

    scaler = joblib.load(SCALER_FILE)

    # Kembalikan data training ke skala asli.
    train_raw = pd.DataFrame(
        scaler.inverse_transform(
            train_df[FEATURES]
        ),
        columns=FEATURES
    )

    # Statistik distribusi training.
    print("=" * 55)
    print("          PANOPTIX DATA DISTRIBUTION AUDIT")
    print("=" * 55)

    print("\nTRAINING DATA RANGE")

    print(
        train_raw[FEATURES].describe(
            percentiles=[0.05, 0.25, 0.50, 0.75, 0.95]
        ).round(2)
    )

    # Batas minimum dan maksimum dari data training.
    cpu_min = train_raw["cpu_usage"].min()
    cpu_max = train_raw["cpu_usage"].max()

    ram_min = train_raw["ram_usage"].min()
    ram_max = train_raw["ram_usage"].max()

    # Identifikasi area grid dalam rentang min-max training.
    inside_range = (
        boundary_df["cpu_usage"].between(cpu_min, cpu_max)
        & boundary_df["ram_usage"].between(ram_min, ram_max)
    )

    inside_df = boundary_df[inside_range]

    outside_df = boundary_df[~inside_range]

    print("\nOBSERVED TRAINING RANGE")

    print(f"CPU : {cpu_min:.2f}% - {cpu_max:.2f}%")
    print(f"RAM : {ram_min:.2f}% - {ram_max:.2f}%")

    print("\nBOUNDARY CONSISTENCY")

    for name, subset in [
        ("Seluruh kombinasi", boundary_df),
        ("Dalam rentang min-max training", inside_df),
        ("Di luar rentang training", outside_df)
    ]:

        if len(subset) == 0:
            continue

        consistent = int(
            subset["is_consistent"].sum()
        )

        total = len(subset)

        percentage = consistent / total * 100

        print(f"\n{name}")
        print(f"Total       : {total}")
        print(f"Konsisten   : {consistent}")
        print(f"Tidak cocok : {total - consistent}")
        print(f"Consistency : {percentage:.2f}%")

    # Kesalahan pada rentang min-max training.
    mismatch_inside = inside_df[
        ~inside_df["is_consistent"]
    ]

    print("\nCONTOH KETIDAKSESUAIAN DALAM RENTANG TRAINING")

    if mismatch_inside.empty:

        print("Tidak ditemukan.")

    else:

        print(
            mismatch_inside[
                [
                    "cpu_usage",
                    "ram_usage",
                    "system_load_score",
                    "rule_label",
                    "ann_prediction",
                    "confidence"
                ]
            ].head(20).to_string(index=False)
        )

    print("\n" + "=" * 55)
    print("              AUDIT COMPLETED")
    print("=" * 55)


if __name__ == "__main__":
    main()
