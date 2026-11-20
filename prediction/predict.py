
from pathlib import Path
import json

import joblib
import numpy as np
import tensorflow as tf
import pandas as pd


# ==========================================
# PATH CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

PROCESSED_DIR = BASE_DIR / "dataset" / "processed"
MODEL_DIR = BASE_DIR / "model"

MODEL_FILE = MODEL_DIR / "panoptix_ann.keras"
SCALER_FILE = PROCESSED_DIR / "scaler.pkl"
CLASS_FILE = MODEL_DIR / "class_mapping.json"

FEATURES = ["cpu_usage", "ram_usage"]


# ==========================================
# PREDICTION CLASS
# ==========================================

class PanoptixPredictor:

    def __init__(self):

        # Memastikan file tersedia
        for file_path in [
            MODEL_FILE,
            SCALER_FILE,
            CLASS_FILE
        ]:

            if not file_path.exists():
                raise FileNotFoundError(
                    f"File tidak ditemukan: {file_path}"
                )

        # Memuat model ANN
        self.model = tf.keras.models.load_model(
            MODEL_FILE
        )

        # Memuat scaler hasil preprocessing
        self.scaler = joblib.load(
            SCALER_FILE
        )

        # Memuat pemetaan kelas
        with open(
            CLASS_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            self.class_mapping = json.load(file)

        print("Panoptix Predictor berhasil dimuat.")

    # ======================================
    # PREDICT SYSTEM LOAD
    # ======================================

    def predict(self, cpu_usage, ram_usage):

        # Validasi input
        values = np.array(
            [cpu_usage, ram_usage],
            dtype=np.float64
        )

        if not np.isfinite(values).all():
            raise ValueError(
                "CPU dan RAM harus berupa angka valid."
            )

        if not np.all((values >= 0) & (values <= 100)):
            raise ValueError(
                "CPU dan RAM harus berada pada rentang 0-100%."
            )

        # Membentuk data input
        input_data = pd.DataFrame(
            [[cpu_usage, ram_usage]],
            columns=FEATURES
        )
        

        # Scaling menggunakan scaler yang sama
        scaled_data = self.scaler.transform(
            input_data
        )

        # Prediksi menggunakan ANN
        probabilities = self.model.predict(
            scaled_data,
            verbose=0
        )[0]

        # Mengambil kelas dengan probabilitas tertinggi
        predicted_index = int(
            np.argmax(probabilities)
        )

        predicted_label = self.class_mapping[
            str(predicted_index)
        ]

        confidence = float(
            probabilities[predicted_index]
        )

        # Menyusun probabilitas setiap kelas
        class_probabilities = {
            self.class_mapping[str(index)]: float(probability)
            for index, probability in enumerate(probabilities)
        }

        return {
            "cpu_usage": float(cpu_usage),
            "ram_usage": float(ram_usage),
            "predicted_class": predicted_label,
            "confidence": confidence,
            "probabilities": class_probabilities
        }


# ==========================================
# TEST PREDICTION
# ==========================================

if __name__ == "__main__":

    predictor = PanoptixPredictor()

    # Contoh input pengujian
    result = predictor.predict(
        cpu_usage=65.0,
        ram_usage=75.0
    )

    print("\n=== PANOPTIX PREDICTION RESULT ===")

    print(f"CPU Usage : {result['cpu_usage']:.2f}%")
    print(f"RAM Usage : {result['ram_usage']:.2f}%")

    print(
        f"Prediction: {result['predicted_class']}"
    )

    print(
        f"Confidence: {result['confidence'] * 100:.2f}%"
    )

    print("\nClass Probabilities:")

    for label, probability in result["probabilities"].items():

        print(
            f"{label:<10}: {probability * 100:.2f}%"
        )
