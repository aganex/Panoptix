
from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent.parent

PROCESSED_DIR = BASE_DIR / "dataset" / "processed"
EVALUATION_DIR = BASE_DIR / "evaluation"

TRAIN_FILE = PROCESSED_DIR / "train.csv"
SCALER_FILE = PROCESSED_DIR / "scaler.pkl"

OUTPUT_FILE = EVALUATION_DIR / "training_distribution.png"

FEATURES = ["cpu_usage", "ram_usage"]

CLASS_NAMES = {
    0: "Low",
    1: "Moderate",
    2: "High"
}


def main():

    df = pd.read_csv(TRAIN_FILE)
    scaler = joblib.load(SCALER_FILE)

    # Kembalikan fitur ke persentase asli.
    raw_features = scaler.inverse_transform(
        df[FEATURES]
    )

    df[FEATURES] = raw_features

    df["category"] = df["label"].map(CLASS_NAMES)

    fig, ax = plt.subplots(figsize=(10, 7))

    # Scatter plot data aktual.
    for label, name in CLASS_NAMES.items():

        subset = df[df["label"] == label]

        ax.scatter(
            subset["cpu_usage"],
            subset["ram_usage"],
            label=name,
            alpha=0.65,
            s=35
        )

    # Garis batas skor 40 dan 70.
    cpu_values = range(101)

    for threshold, style in [
        (40, "--"),
        (70, "-.")
    ]:

        ram_boundary = [
            (threshold - 0.6 * cpu) / 0.4
            for cpu in cpu_values
        ]

        ax.plot(
            cpu_values,
            ram_boundary,
            linestyle=style,
            linewidth=2,
            label=f"Score boundary {threshold}"
        )

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    ax.set_title(
        "Panoptix - Training Data Distribution"
    )

    ax.set_xlabel("CPU Usage (%)")
    ax.set_ylabel("RAM Usage (%)")

    ax.grid(alpha=0.25)
    ax.legend()

    plt.tight_layout()

    EVALUATION_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.savefig(
        OUTPUT_FILE,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("=" * 50)
    print("TRAINING DISTRIBUTION VISUALIZATION")
    print("=" * 50)

    print(f"Jumlah data : {len(df)}")
    print(f"Output      : {OUTPUT_FILE}")

    print("\nDistribusi kelas:")
    print(df["category"].value_counts())

    print("\nVisualisasi berhasil dibuat.")


if __name__ == "__main__":
    main()
