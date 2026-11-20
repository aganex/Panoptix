
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)


# ==========================================
# PATH CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

PROCESSED_DIR = BASE_DIR / "dataset" / "processed"
MODEL_DIR = BASE_DIR / "model"
EVALUATION_DIR = BASE_DIR / "evaluation"

TEST_FILE = PROCESSED_DIR / "test.csv"
SCALER_FILE = PROCESSED_DIR / "scaler.pkl"
MODEL_FILE = MODEL_DIR / "panoptix_ann.keras"

TEST_RESULT_FILE = EVALUATION_DIR / "consistency_test_results.csv"
BOUNDARY_RESULT_FILE = EVALUATION_DIR / "consistency_boundary_results.csv"
SUMMARY_FILE = EVALUATION_DIR / "consistency_validation_summary.json"


# ==========================================
# CONFIGURATION
# ==========================================

FEATURES = ["cpu_usage", "ram_usage"]

CLASS_NAMES = ["Low", "Moderate", "High"]

CLASS_LABELS = [0, 1, 2]


# ==========================================
# RULE-BASED CLASSIFICATION
# ==========================================

def calculate_rule_score(cpu, ram):

    return (0.6 * cpu) + (0.4 * ram)


def classify_rule(score):

    if score < 40:
        return 0

    elif score < 70:
        return 1

    return 2


# ==========================================
# LOAD COMPONENTS
# ==========================================

def load_components():

    for file_path in [TEST_FILE, SCALER_FILE, MODEL_FILE]:

        if not file_path.exists():
            raise FileNotFoundError(
                f"File tidak ditemukan: {file_path}"
            )

    test_df = pd.read_csv(TEST_FILE)

    required_columns = FEATURES + ["label"]

    missing = [
        col for col in required_columns
        if col not in test_df.columns
    ]

    if missing:
        raise ValueError(
            f"Kolom test.csv tidak lengkap: {missing}"
        )

    scaler = joblib.load(SCALER_FILE)

    model = tf.keras.models.load_model(MODEL_FILE)

    return test_df, scaler, model


# ==========================================
# PREPARE TEST DATA
# ==========================================

def prepare_test_data(test_df, scaler):

    # Data pada test.csv sudah dalam kondisi scaled.
    X_scaled = test_df[FEATURES].to_numpy(
        dtype=np.float32
    )

    y_actual = test_df["label"].to_numpy(
        dtype=np.int32
    )

    if not np.isfinite(X_scaled).all():
        raise ValueError(
            "Fitur testing mengandung nilai tidak valid."
        )

    if not np.isin(y_actual, CLASS_LABELS).all():
        raise ValueError(
            "Label testing harus berupa 0, 1, atau 2."
        )

    # Kembalikan fitur ke skala CPU/RAM asli.
    X_raw = scaler.inverse_transform(X_scaled)

    raw_df = pd.DataFrame(
        X_raw,
        columns=FEATURES
    )

    return X_scaled, raw_df, y_actual


# ==========================================
# VALIDATE TEST DATA
# ==========================================

def validate_test_data(
    X_scaled,
    raw_df,
    y_actual,
    model
):

    probabilities = model.predict(
        X_scaled,
        verbose=0
    )

    y_prediction = np.argmax(
        probabilities,
        axis=1
    )

    confidence = np.max(
        probabilities,
        axis=1
    )

    # Hitung hasil klasifikasi berdasarkan formula.
    scores = calculate_rule_score(
        raw_df["cpu_usage"].to_numpy(),
        raw_df["ram_usage"].to_numpy()
    )

    y_rule = np.array([
        classify_rule(score)
        for score in scores
    ])

    results = pd.DataFrame({

        "cpu_usage": raw_df["cpu_usage"],
        "ram_usage": raw_df["ram_usage"],
        "system_load_score": scores,

        "actual_label": [
            CLASS_NAMES[i] for i in y_actual
        ],

        "rule_label": [
            CLASS_NAMES[i] for i in y_rule
        ],

        "ann_prediction": [
            CLASS_NAMES[i] for i in y_prediction
        ],

        "confidence": confidence,

        "rule_match": y_rule == y_prediction,

        "actual_match": y_actual == y_prediction

    })

    results.to_csv(
        TEST_RESULT_FILE,
        index=False
    )

    # Evaluasi konsistensi.
    rule_accuracy = accuracy_score(
        y_rule,
        y_prediction
    )

    actual_accuracy = accuracy_score(
        y_actual,
        y_prediction
    )

    rule_mismatch = int(
        np.sum(y_rule != y_prediction)
    )

    actual_mismatch = int(
        np.sum(y_actual != y_prediction)
    )

    print("\n" + "=" * 55)
    print("       TEST DATA CONSISTENCY VALIDATION")
    print("=" * 55)

    print(f"Jumlah data testing       : {len(y_actual)}")

    print(
        f"ANN vs Rule Accuracy      : {rule_accuracy * 100:.2f}%"
    )

    print(
        f"ANN vs Actual Accuracy    : {actual_accuracy * 100:.2f}%"
    )

    print(f"ANN berbeda dari Rule     : {rule_mismatch}")
    print(f"ANN berbeda dari Actual   : {actual_mismatch}")

    print("\nConfusion Matrix ANN vs Rule:")

    print(
        confusion_matrix(
            y_rule,
            y_prediction,
            labels=CLASS_LABELS
        )
    )

    print("\nClassification Report ANN vs Rule:")

    print(
        classification_report(
            y_rule,
            y_prediction,
            labels=CLASS_LABELS,
            target_names=CLASS_NAMES,
            zero_division=0
        )
    )

    return rule_accuracy, actual_accuracy, rule_mismatch


# ==========================================
# VALIDATE DECISION BOUNDARIES
# ==========================================

def validate_boundaries(scaler, model):

    print("\nMenguji kombinasi CPU dan RAM...")

    # Seluruh kombinasi CPU/RAM dari 0 sampai 100.
    values = np.arange(0, 101, 1)

    cpu_grid, ram_grid = np.meshgrid(
        values,
        values
    )

    raw_input = pd.DataFrame({

        "cpu_usage": cpu_grid.flatten(),
        "ram_usage": ram_grid.flatten()

    })

    # Hitung kategori berdasarkan formula.
    scores = calculate_rule_score(
        raw_input["cpu_usage"].to_numpy(),
        raw_input["ram_usage"].to_numpy()
    )

    y_rule = np.array([
        classify_rule(score)
        for score in scores
    ])

    # Transformasi menggunakan scaler training.
    X_scaled = scaler.transform(raw_input)

    # Prediksi ANN.
    probabilities = model.predict(
        X_scaled,
        verbose=0
    )

    y_prediction = np.argmax(
        probabilities,
        axis=1
    )

    confidence = np.max(
        probabilities,
        axis=1
    )

    boundary_df = raw_input.copy()

    boundary_df["system_load_score"] = scores

    boundary_df["rule_label"] = [
        CLASS_NAMES[i] for i in y_rule
    ]

    boundary_df["ann_prediction"] = [
        CLASS_NAMES[i] for i in y_prediction
    ]

    boundary_df["confidence"] = confidence

    boundary_df["is_consistent"] = (
        y_rule == y_prediction
    )

    boundary_df.to_csv(
        BOUNDARY_RESULT_FILE,
        index=False
    )

    total = len(boundary_df)

    mismatch = int(
        np.sum(y_rule != y_prediction)
    )

    consistency = (
        (total - mismatch) / total
    ) * 100

    print("\n" + "=" * 55)
    print("       DECISION BOUNDARY VALIDATION")
    print("=" * 55)

    print(f"Total kombinasi CPU/RAM : {total}")
    print(f"Prediksi konsisten      : {total - mismatch}")
    print(f"Prediksi tidak konsisten: {mismatch}")
    print(f"Consistency              : {consistency:.2f}%")

    if mismatch > 0:

        print("\nContoh prediksi yang tidak konsisten:")

        print(
            boundary_df.loc[
                ~boundary_df["is_consistent"]
            ].head(15).to_string(index=False)
        )

    return consistency, mismatch, total


# ==========================================
# MAIN VALIDATION
# ==========================================

def main():

    EVALUATION_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 55)
    print("       PANOPTIX MODEL CONSISTENCY VALIDATION")
    print("=" * 55)

    test_df, scaler, model = load_components()

    X_scaled, raw_df, y_actual = prepare_test_data(
        test_df,
        scaler
    )

    test_rule_accuracy, test_actual_accuracy, test_mismatch = (
        validate_test_data(
            X_scaled,
            raw_df,
            y_actual,
            model
        )
    )

    boundary_accuracy, boundary_mismatch, boundary_total = (
        validate_boundaries(
            scaler,
            model
        )
    )

    summary = {

        "model": MODEL_FILE.name,

        "test_samples": int(len(y_actual)),

        "test_ann_vs_rule_accuracy": float(
            test_rule_accuracy
        ),

        "test_ann_vs_actual_accuracy": float(
            test_actual_accuracy
        ),

        "test_rule_mismatch": int(test_mismatch),

        "boundary_total_combinations": int(
            boundary_total
        ),

        "boundary_consistency": float(
            boundary_accuracy
        ),

        "boundary_mismatch": int(
            boundary_mismatch
        )

    }

    with open(
        SUMMARY_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            summary,
            file,
            indent=4
        )

    print("\n" + "=" * 55)
    print("       VALIDATION COMPLETED")
    print("=" * 55)

    print(f"Test results     : {TEST_RESULT_FILE}")
    print(f"Boundary results : {BOUNDARY_RESULT_FILE}")
    print(f"Summary          : {SUMMARY_FILE}")

    print("=" * 55)


if __name__ == "__main__":

    try:
        main()

    except Exception as error:

        print("\nVALIDATION FAILED")
        print(f"Error: {error}")

        raise
