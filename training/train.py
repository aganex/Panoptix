
from pathlib import Path
import json

import pandas as pd
import numpy as np
import tensorflow as tf


# ==========================================
# PATH CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

PROCESSED_DIR = BASE_DIR / "dataset" / "processed"
MODEL_DIR = BASE_DIR / "model"
TRAINING_DIR = BASE_DIR / "training"

TRAIN_FILE = PROCESSED_DIR / "train.csv"

# Model eksperimen, tidak menimpa baseline
MODEL_FILE = MODEL_DIR / "panoptix_ann_calibrated.keras"

HISTORY_FILE = TRAINING_DIR / "calibrated_training_history.csv"
METADATA_FILE = MODEL_DIR / "calibrated_model_metadata.json"
CLASS_FILE = MODEL_DIR / "class_mapping.json"


# ==========================================
# TRAINING CONFIGURATION
# ==========================================

FEATURES = ["cpu_usage", "ram_usage"]

EPOCHS = 100
BATCH_SIZE = 16
RANDOM_SEED = 42
VALIDATION_SPLIT = 0.2

LEARNING_RATE = 0.001


# ==========================================
# LOAD TRAINING DATA
# ==========================================

def load_training_data():

    if not TRAIN_FILE.exists():
        raise FileNotFoundError(
            f"Training dataset tidak ditemukan: {TRAIN_FILE}"
        )

    df = pd.read_csv(TRAIN_FILE)

    required_columns = FEATURES + ["label"]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(
                f"Kolom {column} tidak ditemukan dalam train.csv"
            )

    X = df[FEATURES].to_numpy(dtype=np.float32)
    y = df["label"].to_numpy(dtype=np.int32)

    if not np.isfinite(X).all():
        raise ValueError(
            "Data fitur mengandung nilai kosong atau tidak valid."
        )

    if not np.isin(y, [0, 1, 2]).all():
        raise ValueError(
            "Label harus berupa 0, 1, atau 2."
        )

    if set(np.unique(y)) != {0, 1, 2}:
        raise ValueError(
            "Dataset harus memiliki ketiga kelas: Low, Moderate, High."
        )

    if len(X) < 30:
        raise ValueError(
            "Jumlah dataset terlalu sedikit untuk training."
        )

    return X, y


# ==========================================
# BUILD ANN MODEL
# ==========================================

def build_model():

    model = tf.keras.Sequential([

        tf.keras.layers.Input(
            shape=(len(FEATURES),)
        ),

        tf.keras.layers.Dense(
            16,
            activation="relu"
        ),

        tf.keras.layers.Dense(
            8,
            activation="relu"
        ),

        tf.keras.layers.Dense(
            3,
            activation="softmax"
        )

    ])

    model.compile(

        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        ),

        loss="sparse_categorical_crossentropy",

        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(
                name="accuracy"
            )
        ]

    )

    return model


# ==========================================
# TRAIN MODEL
# ==========================================

def train_model():

    tf.keras.utils.set_random_seed(RANDOM_SEED)

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    TRAINING_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    X_train, y_train = load_training_data()

    print("=" * 50)
    print("       PANOPTIX ANN CALIBRATION")
    print("=" * 50)

    print(f"Jumlah data training : {len(X_train)}")
    print(f"Jumlah fitur         : {X_train.shape[1]}")

    print("\nDistribusi label:")

    label_counts = np.bincount(
        y_train,
        minlength=3
    )

    class_names = ["Low", "Moderate", "High"]

    for index, name in enumerate(class_names):
        print(f"{name:<10}: {label_counts[index]}")

    # ======================================
    # BUILD MODEL
    # ======================================

    model = build_model()

    print("\nArsitektur ANN:")
    model.summary()

    # ======================================
    # CALLBACK CONFIGURATION
    # ======================================

    early_stopping = tf.keras.callbacks.EarlyStopping(

        monitor="val_loss",

        patience=15,

        min_delta=0.0001,

        restore_best_weights=True,

        verbose=1

    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(

        monitor="val_loss",

        factor=0.5,

        patience=5,

        min_lr=0.00001,

        verbose=1

    )

    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(

        filepath=MODEL_FILE,

        monitor="val_loss",

        save_best_only=True,

        verbose=1

    )

    # ======================================
    # MODEL TRAINING
    # ======================================

    print("\nMemulai training tanpa class_weight...")
    print("Eksperimen: Baseline vs Calibrated Model\n")

    history = model.fit(

        X_train,

        y_train,

        epochs=EPOCHS,

        batch_size=BATCH_SIZE,

        validation_split=VALIDATION_SPLIT,

        callbacks=[
            early_stopping,
            reduce_lr,
            model_checkpoint
        ],

        shuffle=True,

        verbose=1

    )

    # ======================================
    # LOAD BEST MODEL
    # ======================================

    best_model = tf.keras.models.load_model(
        MODEL_FILE
    )

    # ======================================
    # SAVE TRAINING HISTORY
    # ======================================

    history_df = pd.DataFrame(history.history)

    history_df.to_csv(
        HISTORY_FILE,
        index=False
    )

    # ======================================
    # SAVE CLASS MAPPING
    # ======================================

    class_mapping = {
        "0": "Low",
        "1": "Moderate",
        "2": "High"
    }

    with open(
        CLASS_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            class_mapping,
            file,
            indent=4
        )

    # ======================================
    # SAVE MODEL METADATA
    # ======================================

    best_epoch = int(
        np.argmin(history.history["val_loss"]) + 1
    )

    metadata = {

        "model_name": "Panoptix ANN Calibrated",

        "architecture": [16, 8, 3],

        "input_features": FEATURES,

        "output_classes": class_mapping,

        "optimizer": "Adam",

        "learning_rate": LEARNING_RATE,

        "loss": "sparse_categorical_crossentropy",

        "class_weight": None,

        "epochs_configured": EPOCHS,

        "epochs_executed": len(history.history["loss"]),

        "best_epoch": best_epoch,

        "batch_size": BATCH_SIZE,

        "validation_split": VALIDATION_SPLIT,

        "random_seed": RANDOM_SEED,

        "training_samples": int(len(X_train)),

        "best_validation_loss": float(
            np.min(history.history["val_loss"])
        ),

        "best_validation_accuracy": float(
            history.history["val_accuracy"][best_epoch - 1]
        )

    }

    with open(
        METADATA_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4
        )

    # ======================================
    # TRAINING SUMMARY
    # ======================================

    print("\n" + "=" * 50)
    print("       CALIBRATION TRAINING SELESAI")
    print("=" * 50)

    print(f"Model              : {MODEL_FILE}")
    print(f"History            : {HISTORY_FILE}")
    print(f"Metadata           : {METADATA_FILE}")

    print(f"\nEpoch dijalankan   : {len(history.history['loss'])}")
    print(f"Best epoch         : {best_epoch}")

    print(
        f"Best training acc  : "
        f"{history.history['accuracy'][best_epoch - 1]:.4f}"
    )

    print(
        f"Best validation acc: "
        f"{history.history['val_accuracy'][best_epoch - 1]:.4f}"
    )

    print(
        f"Best validation loss: "
        f"{np.min(history.history['val_loss']):.6f}"
    )

    print("\nClass weight       : Disabled")

    print("=" * 50)

    return best_model


# ==========================================
# MAIN PROGRAM
# ==========================================

if __name__ == "__main__":
    train_model()
