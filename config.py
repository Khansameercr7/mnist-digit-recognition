"""
============================================================
  Central Configuration
  MNIST Digit Recognition — Arch Technologies
============================================================
"""

import os

# ── Project Structure ──────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "charts")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
FLASK_APP_DIR = os.path.join(BASE_DIR, "flask_app")

# Create directories if they don't exist
for directory in [MODELS_DIR, OUTPUT_DIR, LOGS_DIR]:
    os.makedirs(directory, exist_ok=True)

# ── Model Paths ────────────────────────────────────────────
MODEL_CNN = os.path.join(MODELS_DIR, "cnn_mnist.keras")
MODEL_CNN_TUNED = os.path.join(MODELS_DIR, "cnn_tuned.keras")

# Use tuned model (trained with Keras Tuner + full MNIST dataset)
# Set to True after running: python src/hyperparameter_tuning_keras_tuner.py
USE_TUNED_MODEL = True

# Auto-select best available model
MODEL_PATH = MODEL_CNN_TUNED if (USE_TUNED_MODEL and os.path.exists(MODEL_CNN_TUNED)) else MODEL_CNN

# ── Training Configuration ─────────────────────────────────
SEED = 42
TRAIN_TEST_SPLIT = 0.2
IMAGE_SIZE = 28
IMAGE_CHANNELS = 1
NUM_CLASSES = 10

# CNN Architecture
CNN_PARAMS = {
    "batch_size": 128,
    "epochs": 20,
    "learning_rate": 0.001,
    "dropout_rate": 0.25,
    "conv_filters": [32, 64, 128],
    "dense_units": 256,
}

# Data Augmentation
AUGMENTATION_ENABLED = True
AUGMENTATION_FACTOR = 4  # multiply dataset size
AUGMENTATION_PARAMS = {
    "rotation": 15,  # degrees
    "zoom_range": (0.85, 1.15),
    "shift_range": 2,  # pixels
}

# ── Flask Configuration ────────────────────────────────────
FLASK_DEBUG = False
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000

# ── Logging ────────────────────────────────────────────────
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_LEVEL = "INFO"
