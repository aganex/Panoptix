
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ==========================================
# PATH CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

PROCESSED_DIR = BASE_DIR / "dataset" / "processed"
MODEL_DIR = BASE_DIR / "model"
EVALUATION_DIR = BASE_DIR / "evaluation"

TEST_FILE = PROCESSED_DIR / "test.csv"

MODEL_FILE = MODEL_DIR / "panoptix_ann_calibrated.keras"

REPORT_FILE = EVALUATION_DIR / "calibrated_classification_report.csv"
CONFUSION_MATRIX_FILE = EVALUATION_DIR / "calibrated_confusion_matrix.png"
METRICS_FILE = EVALUATION_DIR / "calibrated_evaluation_metrics.json"
PREDICTIONS_FILE = EVALUATION_DIR / "calibrated_predictions.csv"


# ==========================================
# CLASS CONFIGURATION
# ==========================================

FEATURES = [
    "cpu_usage",
    "ram_usage"
]

CLASS_LABELS = [0, 1, 2]

CLASS_NAMES = [
    "Low",
    "Moderate",
    "High"
]


# ==========================================
# LOAD TEST DATA
# ==========================================

def load_test_data():

    if not TEST_FILE.exists():
        raise FileNotFoundError(
            f"Test dataset tidak ditemukan: {TEST_FILE}"
        )

    df = pd.read_csv(TEST_FILE)

    required_columns = FEATURES + ["label"]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Kolom berikut tidak ditemukan: {missing_columns}"
        )

    X_test = df[FEATURES].to_numpy(dtype=np.float32)
    y_test = df["label"].to_numpy(dtype=np.int32)

    if len(X_test) == 0:
        raise ValueError("Dataset testing kosong.")

    if not np.isfinite(X_test).all():
        raise ValueError(
            "Data testing mengandung NaN atau nilai tidak valid."
        )

    if not np.isin(y_test, CLASS_LABELS).all():
        raise ValueError(
            "Label testing harus berupa 0, 1, atau 2."
        )

    return X_test, y_test


# ==========================================
# LOAD TRAINED MODEL
# ==========================================

def load_model():

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Model tidak ditemukan: {MODEL_FILE}"
        )

    model = tf.keras.models.load_model(
        MODEL_FILE
    )

    if model.output_shape[-1] != len(CLASS_NAMES):
        raise ValueError(
            "Jumlah output model tidak sesuai dengan jumlah kelas."
        )

    return model


# ==========================================
# EVALUATE MODEL
# ==========================================

def evaluate_model():

    EVALUATION_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 50)
    print("          PANOPTIX ANN EVALUATION")
    print("=" * 50)

    # ======================================
    # LOAD DATA AND MODEL
    # ======================================

    print("\nMemuat dataset testing...")

    X_test, y_test = load_test_data()

    print("Memuat calibrated ANN model...")

    model = load_model()

    print(f"\nJumlah data testing : {len(X_test)}")
    print(f"Jumlah fitur        : {X_test.shape[1]}")
    print(f"Jumlah kelas        : {len(CLASS_NAMES)}")

    # ======================================
    # MODEL PREDICTION
    # ======================================

    print("\nMelakukan prediksi...")

    probabilities = model.predict(
        X_test,
        verbose=0
    )

    if not np.isfinite(probabilities).all():
        raise ValueError(
            "Output prediksi model mengandung nilai tidak valid."
        )

    y_pred = np.argmax(
        probabilities,
        axis=1
    )

    confidence = np.max(
        probabilities,
        axis=1
    )

    # ======================================
    # CALCULATE EVALUATION METRICS
    # ======================================

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        labels=CLASS_LABELS,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        labels=CLASS_LABELS,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        labels=CLASS_LABELS,
        average="macro",
        zero_division=0
    )

    correct = int(np.sum(y_test == y_pred))
    incorrect = int(np.sum(y_test != y_pred))

    # ======================================
    # CLASSIFICATION REPORT
    # ======================================

    report = classification_report(
        y_test,
        y_pred,
        labels=CLASS_LABELS,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report).transpose()

    report_df.to_csv(
        REPORT_FILE,
        index=True
    )

    # ======================================
    # CONFUSION MATRIX
    # ======================================

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=CLASS_LABELS
    )

    fig, ax = plt.subplots(
        figsize=(7, 6)
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=CLASS_NAMES
    )

    display.plot(
        ax=ax,
        cmap="Blues",
        values_format="d",
        colorbar=False
    )

    ax.set_title(
        "Panoptix ANN - Calibrated Confusion Matrix"
    )

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")

    plt.tight_layout()

    plt.savefig(
        CONFUSION_MATRIX_FILE,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    # ======================================
    # SAVE INDIVIDUAL PREDICTIONS
    # ======================================

    predictions_df = pd.DataFrame({
        "actual_label": [
            CLASS_NAMES[i] for i in y_test
        ],
        "predicted_label": [
            CLASS_NAMES[i] for i in y_pred
        ],
        "confidence": confidence,
        "correct_prediction": y_test == y_pred
    })

    predictions_df.to_csv(
        PREDICTIONS_FILE,
        index=False
    )

    # ======================================
    # SAVE EVALUATION METRICS
    # ======================================

    metrics = {
        "model": MODEL_FILE.name,
        "accuracy": float(accuracy),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_score_macro": float(f1),
        "total_testing_samples": int(len(y_test)),
        "correct_predictions": correct,
        "incorrect_predictions": incorrect,
        "class_distribution": {
            CLASS_NAMES[i]: int(np.sum(y_test == i))
            for i in CLASS_LABELS
        }
    }

    with open(
        METRICS_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4
        )

    # ======================================
    # DISPLAY RESULTS
    # ======================================

    print("\n" + "=" * 50)
    print("             EVALUATION RESULTS")
    print("=" * 50)

    print(
        f"Accuracy  : {accuracy:.4f} ({accuracy * 100:.2f}%)"
    )

    print(
        f"Precision : {precision:.4f} ({precision * 100:.2f}%)"
    )

    print(
        f"Recall    : {recall:.4f} ({recall * 100:.2f}%)"
    )

    print(
        f"F1-Score  : {f1:.4f} ({f1 * 100:.2f}%)"
    )

    print(f"\nPrediksi benar : {correct}")
    print(f"Prediksi salah : {incorrect}")

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            y_pred,
            labels=CLASS_LABELS,
            target_names=CLASS_NAMES,
            zero_division=0
        )
    )

    print("\nConfusion Matrix:")
    print(cm)

    # ======================================
    # OUTPUT FILES
    # ======================================

    print("\n" + "=" * 50)
    print("           EVALUATION COMPLETED")
    print("=" * 50)

    print(f"Classification report : {REPORT_FILE}")
    print(f"Confusion matrix      : {CONFUSION_MATRIX_FILE}")
    print(f"Evaluation metrics    : {METRICS_FILE}")
    print(f"Predictions           : {PREDICTIONS_FILE}")

    print("=" * 50)


# ==========================================
# MAIN PROGRAM
# ==========================================

if __name__ == "__main__":

    try:
        evaluate_model()

    except Exception as error:

        print("\nEVALUATION FAILED")
        print(f"Error: {error}")

        raise
