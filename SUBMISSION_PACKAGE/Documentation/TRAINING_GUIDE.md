# 🎓 Training Guide — MNIST Model Optimization

This guide explains how to train and optimize the MNIST CNN model using the new Tier 1 improvements.

---

## 📋 Overview of Training Scripts

| Script | Dataset | Method | Expected Accuracy | Time |
|--------|---------|--------|-------------------|------|
| `src/baseline.py` | Sklearn digits (1,797) | Logistic Reg, ANN, MLP | 95-99.9% | ~1 min |
| `src/cnn_full_dataset.py` | Full MNIST (70,000) | Standard CNN | 99.5%+ | ~2 min |
| `src/hyperparameter_tuning_keras_tuner.py` | Full MNIST (70,000) | **Keras Tuner (Bayesian)** | **99.6%+** | ~10-15 min |

---

## 🚀 Step-by-Step Training

### Prerequisites
```bash
pip install -r requirements.txt
```

This installs:
- TensorFlow 2.12+
- Keras Tuner 1.4+ ← **NEW**
- All other dependencies

### Phase 1: Baseline Models (Optional)
Quick sanity check with traditional ML models:
```bash
python src/baseline.py
```
**Output:** `models/`, `outputs/charts/`, benchmark accuracy

---

### Phase 2: CNN Training (Optional)
Basic CNN on full MNIST dataset:
```bash
python src/cnn_full_dataset.py
```

**Expected output:**
- Test accuracy: **99.5%+**
- Trained model: `models/cnn_mnist.keras`
- Visualizations: `outputs/charts/`

---

### ⭐ Phase 3: Keras Tuner Optimization (RECOMMENDED)

This is the **flagship script** combining full dataset + automated optimization:

```bash
python src/hyperparameter_tuning_keras_tuner.py
```

#### What It Does:
1. **Loads full MNIST** (70,000 samples: 60K train + 10K test)
2. **Runs Bayesian Optimization** to find best hyperparameters
   - Tests 20 different configurations automatically
   - Each trial: 10 epochs with early stopping
   - Intelligently explores learning rates, dropout, filter sizes
3. **Rebuilds best model** with full 20 epochs of training
4. **Evaluates on test set** and saves to `models/cnn_tuned.keras`

#### Expected Results:
- **Test Accuracy:** 99.6% or higher
- **Training Time:** 10-15 minutes (CPU), ~2-3 minutes (GPU)
- **Output Files:**
  - `models/cnn_tuned.keras` ← Best model
  - `outputs/charts/keras_tuner_training_history.png`
  - `outputs/charts/keras_tuner_confusion_matrix.png`
  - `outputs/charts/keras_tuner_class_accuracy.png`

---

## 🎯 Use Tuned Model in Web App

After training with Keras Tuner, enable the tuned model:

### Option 1: Edit config.py
```python
# In config.py, line 24:
USE_TUNED_MODEL = True
```

### Option 2: Set Environment Variable
```bash
export MNIST_USE_TUNED=1
python run.py
```

### Verify Model in Web App
```bash
python run.py
```

Visit **http://localhost:5000** and draw digits. The web app will now use the optimized model with ~99.6% accuracy!

---

## 📊 Understanding the Hyperparameter Search

### Tuned Parameters (Keras Tuner Search Space)

| Parameter | Search Range | Default | Notes |
|-----------|--------------|---------|-------|
| **Learning Rate** | 1e-5 to 1e-2 | ~0.0003 | Adam optimizer lr |
| **Dropout Rate** | 0.1 to 0.5 | ~0.25 | Applied after conv/dense |
| **Conv1 Filters** | 16 to 128 | ~64 | First convolution block |
| **Conv2 Filters** | 32 to 256 | ~128 | Second convolution block |
| **Dense Units** | 64 to 512 | ~256 | Fully-connected layer |

### Bayesian Optimization Strategy
- **Algorithm:** Gaussian Process Bayesian Optimization
- **Trials:** 20 maximum (stops early if no improvement)
- **Objective:** Maximize validation accuracy
- **Early stopping:** Patience of 3 epochs per trial

---

## 📈 Comparing Results

### Before (Manual Grid Search on Augmented Data)
```
Dataset:    1,797 samples × augmentation → ~20K samples
Tuning:     Manual grid search (9 combinations)
Accuracy:   87.2%
Time:       ~2 minutes
```

### After (Keras Tuner on Full MNIST)
```
Dataset:    Full MNIST, 70,000 official samples
Tuning:     Bayesian optimization (20 trials)
Accuracy:   99.6%+
Time:       10-15 minutes
```

**Improvement: +12.4% accuracy with better tuning methodology**

---

## 🔧 Advanced Configuration

### Adjust Search Parameters

Edit `src/hyperparameter_tuning_keras_tuner.py`:

```python
MAX_TRIALS = 20              # ← Increase for more thorough search
TUNING_EPOCHS = 10           # ← More epochs per trial = slower but better
EXECUTIONS_PER_TRIAL = 1     # ← Multiple runs of same hyperparams
```

### Custom Hyperparameter Ranges

Modify the `build_model()` function:

```python
def build_model(hp):
    # Change learning rate range:
    lr = hp.Float("learning_rate", 
                   min_value=1e-4,    # ← Adjust minimum
                   max_value=1e-3,    # ← Adjust maximum
                   sampling="log")
    
    # Change dropout range:
    dropout = hp.Float("dropout", 
                        min_value=0.15,  # ← Higher minimum
                        max_value=0.4,   # ← Lower maximum
                        step=0.05)
```

---

## 🐛 Troubleshooting

### Out of Memory (OOM)
```python
# In hyperparameter_tuning_keras_tuner.py, reduce:
batch_size=64  # ← Was 128
TUNING_EPOCHS=5  # ← Was 10
MAX_TRIALS=10  # ← Was 20
```

### Model Not Found When Starting Web App
```bash
# Train the model first:
python src/hyperparameter_tuning_keras_tuner.py

# Then run the app:
python run.py
```

### Slow GPU/CPU Utilization
- Ensure TensorFlow is installed correctly: `pip install tensorflow`
- For GPU: `pip install tensorflow[and-cuda]` (if CUDA available)
- Check: `python -c "import tensorflow as tf; print(tf.config.list_physical_devices())"`

---

## 📝 Next Steps (Tier 2+)

After achieving 99.6%+ accuracy with Keras Tuner:

1. **Ensemble Methods** — Combine multiple trained models
2. **Advanced Data Augmentation** — RandAugment, MixUp, CutMix
3. **Transfer Learning** — Fine-tune pre-trained models (ResNet, EfficientNet)
4. **Model Compression** — Quantization & pruning for faster inference
5. **Adversarial Training** — Robustness against noisy inputs

---

## 📚 References

- [Keras Tuner Documentation](https://keras.io/keras_tuner/)
- [TensorFlow MNIST Dataset](https://www.tensorflow.org/datasets/catalog/mnist)
- [Bayesian Optimization](https://arxiv.org/abs/1507.02188)

