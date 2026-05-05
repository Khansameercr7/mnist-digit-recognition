"""
============================================================
  PHASE 2 — Convolutional Neural Network (CNN)
  MNIST Digit Recognition — Arch Technologies
============================================================
  Architecture:
    Input (28×28×1)
    → Conv2D(32) → BN → MaxPool → Dropout(0.25)
    → Conv2D(64) → BN → MaxPool → Dropout(0.25)
    → Conv2D(128) → BN → Dropout(0.25)
    → Flatten → Dense(256) → Dropout(0.4)
    → Dense(10, Softmax)

  Run: python phase2_cnn.py
============================================================
"""

import os, warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.ndimage import zoom, rotate, shift as nd_shift
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, ConfusionMatrixDisplay, f1_score)

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.utils import to_categorical

SEED = 42
np.random.seed(SEED); tf.random.set_seed(SEED)

SRC_DIR    = os.path.dirname(__file__)
OUTPUT_DIR = os.path.join(SRC_DIR, "..", "outputs", "charts")
MODEL_DIR  = os.path.join(SRC_DIR, "..", "models")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR,  exist_ok=True)

def save(name):
    plt.savefig(os.path.join(OUTPUT_DIR, name), dpi=150, bbox_inches="tight")
    plt.close(); print(f"  ✔  {name}")

# ─── Dataset ────────────────────────────────────────────────
print("\n" + "="*60)
print("  PHASE 2 — CNN Training")
print("="*60)
print("\n[1]  Building augmented dataset …")

digits_sk = load_digits()
X_base = digits_sk.data.astype(np.float32)
y_base = digits_sk.target.astype(np.int32)
X_28   = np.array([zoom(x.reshape(8, 8), 28/8).flatten() for x in X_base])

TARGET = 20_000
rng    = np.random.default_rng(SEED)
aug_imgs, aug_lbls = [X_28.copy()], [y_base.copy()]
for _ in range(TARGET // len(X_28) + 1):
    noise  = rng.normal(0, 0.03, X_28.shape).astype(np.float32)
    angles = rng.uniform(-12, 12, len(X_28))
    shifts = rng.uniform(-1.5, 1.5, (len(X_28), 2))
    batch  = [nd_shift(rotate(x.reshape(28,28), a, reshape=False, cval=0), s, cval=0).flatten()
              for x, a, s in zip(X_28, angles, shifts)]
    aug_imgs.append(np.array(batch, dtype=np.float32) + noise)
    aug_lbls.append(y_base.copy())

X_all = np.clip(np.vstack(aug_imgs)[:TARGET], 0, None)
y_all = np.concatenate(aug_lbls)[:TARGET]
X_all = (X_all / X_all.max()).astype(np.float32)

X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.2, random_state=SEED, stratify=y_all)

X_tr_cnn = X_train.reshape(-1, 28, 28, 1)
X_te_cnn = X_test.reshape(-1, 28, 28, 1)
y_tr_cat  = to_categorical(y_train, 10)
y_te_cat  = to_categorical(y_test,  10)
print(f"       Train: {len(X_train):,}  |  Test: {len(X_test):,}")

# ─── Build CNN ──────────────────────────────────────────────
print("\n[2]  Building CNN …")

def build_cnn(filters1=32, filters2=64, filters3=128,
              dense_units=256, dropout=0.25, lr=0.001):
    m = models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        # Block 1
        layers.Conv2D(filters1, (3,3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)), layers.Dropout(dropout),
        # Block 2
        layers.Conv2D(filters2, (3,3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)), layers.Dropout(dropout),
        # Block 3
        layers.Conv2D(filters3, (3,3), padding="same", activation="relu"),
        layers.BatchNormalization(), layers.Dropout(dropout),
        # Head
        layers.Flatten(),
        layers.Dense(dense_units, activation="relu"),
        layers.Dropout(dropout + 0.15),
        layers.Dense(10, activation="softmax"),
    ], name="MNIST_CNN")
    m.compile(optimizer=tf.keras.optimizers.Adam(lr),
              loss="categorical_crossentropy", metrics=["accuracy"])
    return m

cnn = build_cnn()
cnn.summary()
print(f"\n  Total parameters: {cnn.count_params():,}")

# Architecture table
print("\n[3]  Architecture diagram …")
info = [{"Layer": l.name, "Type": l.__class__.__name__,
          "Output Shape": str(l.output_shape), "Params": f"{l.count_params():,}"}
        for l in cnn.layers]
df_arch = pd.DataFrame(info)
clrs = {"Conv2D":"#D6E4F0","BatchNormalization":"#E8F5E9","MaxPooling2D":"#FFF9C4",
        "Dropout":"#FCE4EC","Dense":"#F3E5F5","Flatten":"#FFF3E0"}
fig, ax = plt.subplots(figsize=(13, 8)); ax.axis("off")
t = ax.table(cellText=df_arch.values, colLabels=df_arch.columns, cellLoc="center", loc="center")
t.auto_set_font_size(False); t.set_fontsize(9); t.scale(1, 1.7)
for j in range(4): t[0,j].set_facecolor("#2E75B6"); t[0,j].set_text_props(color="white", fontweight="bold")
for i, row in df_arch.iterrows():
    bg = clrs.get(row["Type"], "#F2F2F2")
    for j in range(4): t[i+1,j].set_facecolor(bg)
ax.set_title("CNN Architecture — Layer Details", fontsize=14, fontweight="bold",
             color="#1F4E79", pad=20)
plt.tight_layout(); save("CNN1_architecture_table.png")

# ─── Train ──────────────────────────────────────────────────
print("\n[4]  Training CNN (up to 20 epochs + early stopping) …")
cb = [
    callbacks.EarlyStopping("val_accuracy", patience=4, restore_best_weights=True, verbose=0),
    callbacks.ReduceLROnPlateau("val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=0),
]
history = cnn.fit(X_tr_cnn, y_tr_cat, epochs=20, batch_size=64,
                  validation_split=0.15, callbacks=cb, verbose=1)

# ─── Evaluate ───────────────────────────────────────────────
_, test_acc = cnn.evaluate(X_te_cnn, y_te_cat, verbose=0)
y_pred = np.argmax(cnn.predict(X_te_cnn, verbose=0), axis=1)
wrong  = np.where(y_pred != y_test)[0]
print(f"\n  ✅  CNN Test Accuracy : {test_acc*100:.4f}%")
print(f"  ✅  Misclassified     : {len(wrong)} / {len(y_test)}")

# ─── Charts ─────────────────────────────────────────────────
print("\n[5]  Saving evaluation charts …")
ep = range(1, len(history.history["accuracy"])+1)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("CNN — Training History", fontsize=14, fontweight="bold")
axes[0].plot(ep, history.history["accuracy"],    "b-o", ms=4, label="Train")
axes[0].plot(ep, history.history["val_accuracy"],"r-s", ms=4, label="Val")
axes[0].set_title("Accuracy"); axes[0].legend(); axes[0].grid(True, alpha=0.3)
axes[1].plot(ep, history.history["loss"],    "b-o", ms=4, label="Train")
axes[1].plot(ep, history.history["val_loss"],"r-s", ms=4, label="Val")
axes[1].set_title("Loss"); axes[1].legend(); axes[1].grid(True, alpha=0.3)
plt.tight_layout(); save("CNN3_training_curves.png")

fig, ax = plt.subplots(figsize=(9, 7))
ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred), display_labels=range(10)).plot(
    ax=ax, colorbar=True, cmap="Blues")
ax.set_title(f"CNN — Confusion Matrix | Acc: {test_acc*100:.2f}%", fontsize=13, fontweight="bold")
plt.tight_layout(); save("CNN4_confusion_matrix.png")

f1s = f1_score(y_test, y_pred, average=None)
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(range(10), f1s*100, color=[plt.cm.RdYlGn(max(0,(v-0.7)/0.3)) for v in f1s], edgecolor="black")
ax.set_xticks(range(10)); ax.set_xticklabels([f"Digit {d}" for d in range(10)])
ax.set_title("CNN — Per-class F1-Score", fontsize=13, fontweight="bold"); ax.set_ylabel("F1-Score (%)")
[ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.5, f"{f*100:.1f}%", ha="center", fontsize=9, fontweight="bold")
 for b, f in zip(bars, f1s)]
plt.tight_layout(); save("CNN5_f1_per_class.png")

np.random.seed(7); idx = np.random.choice(len(X_test), 20, replace=False)
proba = cnn.predict(X_te_cnn, verbose=0)
fig = plt.figure(figsize=(20, 5))
fig.suptitle("CNN — Sample Predictions", fontsize=13, fontweight="bold")
for i, si in enumerate(idx):
    ax = fig.add_subplot(2, 10, i+1)
    ax.imshow(X_test[si].reshape(28, 28), cmap="gray")
    p, t = y_pred[si], y_test[si]
    ax.set_title(f"P:{p} T:{t}\n{proba[si][p]*100:.0f}%", fontsize=7,
                 color="green" if p==t else "red", fontweight="bold"); ax.axis("off")
plt.tight_layout(); save("CNN6_sample_predictions.png")

print(f"\n[6]  Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=[str(d) for d in range(10)]))

# ─── Save model ─────────────────────────────────────────────
cnn.save(os.path.join(MODEL_DIR, "cnn_mnist.keras"))
np.save(os.path.join(MODEL_DIR, "cnn_acc.npy"), np.array([test_acc]))
print(f"\n  💾  Model saved → {MODEL_DIR}/cnn_mnist.keras")
print("\n" + "="*60)
print(f"  PHASE 2 COMPLETE  |  CNN Accuracy: {test_acc*100:.4f}%")
print("="*60 + "\n")
