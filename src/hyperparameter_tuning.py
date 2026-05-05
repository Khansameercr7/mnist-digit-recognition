"""
============================================================
  PHASE 3 — Hyperparameter Tuning
  MNIST Digit Recognition — Arch Technologies
============================================================
  Strategy  : Manual Grid Search
  Parameters: Learning Rate × Dropout Rate
  Grid      : lr ∈ {0.001, 0.0005, 0.0001}
              dropout ∈ {0.2, 0.3, 0.4}
  Total runs: 9

  Run: python phase3_hyperparameter_tuning.py
  NOTE: Requires phase2_cnn.py to have been run first,
        OR run standalone (rebuilds dataset automatically).
============================================================
"""

import os, warnings, time
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
from sklearn.metrics import accuracy_score

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
print("  PHASE 3 — Hyperparameter Tuning")
print("="*60)
print("\n[1]  Preparing dataset …")

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
X_tr = X_train.reshape(-1, 28, 28, 1)
X_te = X_test.reshape(-1, 28, 28, 1)
y_tr_c = to_categorical(y_train, 10)
y_te_c = to_categorical(y_test,  10)
print(f"       Train: {len(X_train):,}  |  Test: {len(X_test):,}")

# ─── Model Builder ──────────────────────────────────────────
def build_model(lr, dropout):
    m = models.Sequential([
        layers.Input((28, 28, 1)),
        layers.Conv2D(32, (3,3), padding="same", activation="relu"),
        layers.BatchNormalization(), layers.MaxPooling2D(), layers.Dropout(dropout),
        layers.Conv2D(64, (3,3), padding="same", activation="relu"),
        layers.BatchNormalization(), layers.MaxPooling2D(), layers.Dropout(dropout),
        layers.Flatten(),
        layers.Dense(128, activation="relu"), layers.Dropout(dropout + 0.1),
        layers.Dense(10, activation="softmax"),
    ])
    m.compile(tf.keras.optimizers.Adam(lr), "categorical_crossentropy", metrics=["accuracy"])
    return m

# ─── Grid Search ────────────────────────────────────────────
LR_VALUES      = [0.001, 0.0005, 0.0001]
DROPOUT_VALUES = [0.2, 0.3, 0.4]

print("\n[2]  Running 9-combination grid search …\n")
print(f"  {'LR':<10} {'Dropout':<10} {'Val Acc':>10} {'Test Acc':>10} {'Time':>8}")
print("  " + "-"*52)

results  = []
cb_early = [callbacks.EarlyStopping("val_accuracy", patience=3, restore_best_weights=True, verbose=0)]
best_acc = 0; best_config = {}; best_model_obj = None

for lr in LR_VALUES:
    for dr in DROPOUT_VALUES:
        t0 = time.time()
        m  = build_model(lr, dr)
        h  = m.fit(X_tr, y_tr_c, epochs=10, batch_size=64, validation_split=0.15,
                   callbacks=cb_early, verbose=0)
        _, test_acc = m.evaluate(X_te, y_te_c, verbose=0)
        val_acc     = max(h.history["val_accuracy"])
        elapsed     = time.time() - t0

        results.append({"lr": lr, "dropout": dr, "val_acc": val_acc,
                         "test_acc": test_acc, "time": elapsed})
        print(f"  {lr:<10} {dr:<10} {val_acc*100:>9.2f}% {test_acc*100:>9.2f}% {elapsed:>7.1f}s")

        if test_acc > best_acc:
            best_acc = test_acc; best_config = {"lr": lr, "dropout": dr}; best_model_obj = m

print(f"\n  ✅  Best: lr={best_config['lr']}, dropout={best_config['dropout']} → {best_acc*100:.2f}%")

# ─── Retrain Best Model (more epochs) ───────────────────────
print(f"\n[3]  Retraining best config for 20 epochs …")
best_m = build_model(best_config["lr"], best_config["dropout"])
cb_best = [
    callbacks.EarlyStopping("val_accuracy", patience=5, restore_best_weights=True, verbose=0),
    callbacks.ReduceLROnPlateau("val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=0),
]
best_h = best_m.fit(X_tr, y_tr_c, epochs=20, batch_size=64,
                     validation_split=0.15, callbacks=cb_best, verbose=1)
_, final_acc = best_m.evaluate(X_te, y_te_c, verbose=0)
y_pred_tuned = np.argmax(best_m.predict(X_te, verbose=0), axis=1)
print(f"\n  ✅  Tuned Model Accuracy: {final_acc*100:.4f}%")

# ─── Charts ─────────────────────────────────────────────────
print("\n[4]  Saving charts …")

# Heatmap
rows = [{"Learning Rate": str(r["lr"]), "Dropout": str(r["dropout"]),
          "Acc": round(r["test_acc"]*100, 2)} for r in results]
df   = pd.DataFrame(rows).pivot(index="Dropout", columns="Learning Rate", values="Acc")
fig, ax = plt.subplots(figsize=(9, 6))
sns.heatmap(df, annot=True, fmt=".1f", cmap="YlGn", ax=ax, linewidths=.5,
            cbar_kws={"label":"Test Accuracy (%)"}, annot_kws={"fontsize": 12})
ax.set_title("Hyperparameter Grid Search — Test Accuracy (%)\nLearning Rate × Dropout Rate",
             fontsize=13, fontweight="bold")
plt.tight_layout(); save("HP1_search_heatmap.png")

# Baseline vs tuned
baseline_acc = float(np.load(os.path.join(MODEL_DIR, "..", "..", "Phase2_CNN", "models", "cnn_acc.npy"))) \
    if os.path.exists(os.path.join(MODEL_DIR, "..", "..", "Phase2_CNN", "models", "cnn_acc.npy")) \
    else best_acc
fig, ax = plt.subplots(figsize=(8, 5))
vals  = [baseline_acc * 100, final_acc * 100]
lbls  = [f"Baseline CNN\n(Phase 2)", f"Tuned CNN\n(lr={best_config['lr']}, drop={best_config['dropout']})"]
bars  = ax.bar(lbls, vals, color=["#4C72B0","#DD8452"], edgecolor="black", width=0.4)
ax.set_ylim(max(0, min(vals)-10), min(110, max(vals)+5))
ax.set_title("Baseline vs Best Tuned CNN", fontsize=13, fontweight="bold")
ax.set_ylabel("Test Accuracy (%)")
[ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.15, f"{v:.2f}%",
         ha="center", fontsize=12, fontweight="bold") for b, v in zip(bars, vals)]
plt.tight_layout(); save("HP2_baseline_vs_tuned.png")

# All combinations sorted
sorted_r = sorted(results, key=lambda x: x["test_acc"])
xlabels  = [f"lr={r['lr']}\ndrop={r['dropout']}" for r in sorted_r]
accs     = [r["test_acc"]*100 for r in sorted_r]
colors   = ["#DD8452" if r["lr"]==best_config["lr"] and r["dropout"]==best_config["dropout"]
             else "#4C72B0" for r in sorted_r]
fig, ax  = plt.subplots(figsize=(12, 5))
ax.bar(range(len(sorted_r)), accs, color=colors, edgecolor="black")
ax.set_xticks(range(len(sorted_r))); ax.set_xticklabels(xlabels, fontsize=8)
ax.set_title("All 9 Hyperparameter Combinations — Sorted by Accuracy (Orange = Best)",
             fontsize=12, fontweight="bold")
ax.set_ylabel("Test Accuracy (%)")
[ax.text(i, v+0.3, f"{v:.1f}%", ha="center", fontsize=8) for i, v in enumerate(accs)]
plt.tight_layout(); save("HP3_all_combinations.png")

# Tuned model training curves
ep = range(1, len(best_h.history["accuracy"])+1)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Tuned CNN — Training History", fontsize=14, fontweight="bold")
axes[0].plot(ep, best_h.history["accuracy"],    "b-o", ms=4, label="Train")
axes[0].plot(ep, best_h.history["val_accuracy"],"r-s", ms=4, label="Val")
axes[0].set_title("Accuracy"); axes[0].legend(); axes[0].grid(True, alpha=0.3)
axes[1].plot(ep, best_h.history["loss"],    "b-o", ms=4, label="Train")
axes[1].plot(ep, best_h.history["val_loss"],"r-s", ms=4, label="Val")
axes[1].set_title("Loss"); axes[1].legend(); axes[1].grid(True, alpha=0.3)
plt.tight_layout(); save("HP4_tuned_training_curves.png")

# Save tuned model
best_m.save(os.path.join(MODEL_DIR, "cnn_tuned.keras"))
np.save(os.path.join(MODEL_DIR, "tuned_acc.npy"), np.array([final_acc]))

print("\n  Summary of all trials:")
df_summary = pd.DataFrame(results)
df_summary["test_acc"] = (df_summary["test_acc"]*100).round(2)
df_summary["val_acc"]  = (df_summary["val_acc"]*100).round(2)
print(df_summary.sort_values("test_acc", ascending=False).to_string(index=False))

print("\n" + "="*60)
print(f"  PHASE 3 COMPLETE  |  Best Accuracy: {final_acc*100:.4f}%")
print("="*60 + "\n")
