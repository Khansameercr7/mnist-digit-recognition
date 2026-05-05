"""
============================================================
  PHASE 1 — Baseline Machine Learning Models
  MNIST Digit Recognition — Arch Technologies
============================================================
  Models  : Logistic Regression | ANN (MLP) | Deep MLP
  Dataset : MNIST (sklearn digits → augmented to 20,000)
  Run     : python phase1_baseline.py
============================================================
"""

import os, time, warnings
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.ndimage import zoom, rotate, shift as nd_shift
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, ConfusionMatrixDisplay)
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "charts")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def save(name):
    plt.savefig(os.path.join(OUTPUT_DIR, name), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✔  {name}")

# ─── Dataset ────────────────────────────────────────────────
print("\n" + "="*60)
print("  PHASE 1 — Baseline ML Models")
print("="*60)
print("\n[1]  Building dataset (20,000 augmented samples) …")

digits_sk = load_digits()
X_base = digits_sk.data.astype(np.float32)
y_base = digits_sk.target.astype(np.int32)
X_28   = np.array([zoom(x.reshape(8, 8), 28/8).flatten() for x in X_base])

TARGET = 20_000
rng    = np.random.default_rng(42)
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

# Split & scale
X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.2, random_state=42, stratify=y_all)
scaler  = StandardScaler()
X_tr_sc = scaler.fit_transform(X_train)
X_te_sc = scaler.transform(X_test)
print(f"       Train: {len(X_train):,}  |  Test: {len(X_test):,}")

# ─── Class Distribution ─────────────────────────────────────
print("\n[2]  Plotting class distribution …")
unique, counts = np.unique(y_all, return_counts=True)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("MNIST Dataset — Class Distribution", fontsize=14, fontweight="bold")
axes[0].bar(unique, counts, color=plt.cm.tab10(np.linspace(0,1,10)), edgecolor="black")
axes[0].set_xlabel("Digit"); axes[0].set_ylabel("Count"); axes[0].set_title("Sample Count per Class")
axes[1].pie(counts, labels=[str(d) for d in unique], autopct="%1.1f%%",
            colors=plt.cm.tab10(np.linspace(0,1,10)))
axes[1].set_title("Class Proportion")
plt.tight_layout(); save("A1_class_distribution.png")

# ─── Sample Images ──────────────────────────────────────────
print("[3]  Sample images …")
fig, axes = plt.subplots(2, 10, figsize=(18, 4))
fig.suptitle("Sample Images — Two per Digit Class", fontsize=13, fontweight="bold")
for row in range(2):
    for digit in range(10):
        idx = np.where(y_all == digit)[0][row * 5]
        axes[row, digit].imshow(X_all[idx].reshape(28, 28), cmap="gray")
        axes[row, digit].set_title(f"Digit {digit}", fontsize=8)
        axes[row, digit].axis("off")
plt.tight_layout(); save("A2_sample_images.png")

# ─── Normalisation ──────────────────────────────────────────
print("[4]  Normalisation comparison …")
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].hist((X_all * 255).flatten(), bins=50, color="coral", edgecolor="none", alpha=0.85)
axes[0].set_title("Before Normalisation (0–255)")
axes[1].hist(X_all.flatten(), bins=50, color="mediumseagreen", edgecolor="none", alpha=0.85)
axes[1].set_title("After Normalisation (0–1)")
fig.suptitle("Pixel Intensity — Before vs After Normalisation", fontsize=13, fontweight="bold")
plt.tight_layout(); save("B1_normalisation.png")

# ─── PCA ────────────────────────────────────────────────────
print("[5]  PCA 2-D projection …")
pca   = PCA(n_components=2, random_state=42)
sub   = np.random.choice(len(X_tr_sc), 5000, replace=False)
X_pca = pca.fit_transform(X_tr_sc[sub])
y_sub = y_train[sub]
fig, ax = plt.subplots(figsize=(9, 7))
for d in range(10):
    m = y_sub == d
    ax.scatter(X_pca[m, 0], X_pca[m, 1], s=5, alpha=0.5, label=str(d), color=plt.cm.tab10(d/10))
ax.legend(title="Digit", markerscale=3, ncol=2)
ax.set_title("PCA 2-D Projection of Training Data", fontsize=13, fontweight="bold")
plt.tight_layout(); save("B2_pca_2d.png")

# ─── Train Models ───────────────────────────────────────────
print("\n[6]  Training models …")
results = {}

# Logistic Regression
t0  = time.time()
lr  = LogisticRegression(max_iter=100, C=0.1, solver="lbfgs", n_jobs=-1, random_state=42)
lr.fit(X_tr_sc, y_train)
acc = accuracy_score(y_test, lr.predict(X_te_sc))
results["Logistic\nRegression"] = {"acc": acc, "time": time.time()-t0,
                                    "y_pred": lr.predict(X_te_sc), "cm": confusion_matrix(y_test, lr.predict(X_te_sc))}
print(f"  Logistic Regression : {acc*100:.2f}%  ({time.time()-t0:.1f}s)")

# ANN
t0  = time.time()
ann = MLPClassifier(hidden_layer_sizes=(256, 128), activation="relu", solver="adam",
                    max_iter=15, batch_size=256, learning_rate_init=0.001, random_state=42)
ann.fit(X_tr_sc, y_train)
acc = accuracy_score(y_test, ann.predict(X_te_sc))
results["ANN\n(MLP 256-128)"] = {"acc": acc, "time": time.time()-t0,
                                   "y_pred": ann.predict(X_te_sc), "cm": confusion_matrix(y_test, ann.predict(X_te_sc))}
print(f"  ANN (256→128)       : {acc*100:.2f}%  ({time.time()-t0:.1f}s)")

# Deep MLP
t0   = time.time()
deep = MLPClassifier(hidden_layer_sizes=(256, 128, 64), activation="relu", solver="adam",
                     max_iter=25, batch_size=256, learning_rate_init=0.001,
                     early_stopping=True, validation_fraction=0.1, n_iter_no_change=5, random_state=42)
deep.fit(X_tr_sc, y_train)
acc = accuracy_score(y_test, deep.predict(X_te_sc))
results["Deep MLP\n(256-128-64)"] = {"acc": acc, "time": time.time()-t0,
                                      "y_pred": deep.predict(X_te_sc), "cm": confusion_matrix(y_test, deep.predict(X_te_sc))}
print(f"  Deep MLP (256-128-64): {acc*100:.2f}%  ({time.time()-t0:.1f}s)")

# ─── Evaluation Charts ──────────────────────────────────────
print("\n[7]  Evaluation charts …")
model_names = list(results.keys())
accuracies  = [results[m]["acc"] for m in model_names]
times       = [results[m]["time"] for m in model_names]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Model Comparison", fontsize=14, fontweight="bold")
colors = ["#4C72B0", "#DD8452", "#55A868"]
bars   = axes[0].bar(model_names, [a*100 for a in accuracies], color=colors, edgecolor="black", width=0.5)
axes[0].set_ylim(85, 102); axes[0].set_title("Test Accuracy (%)")
[axes[0].text(b.get_x()+b.get_width()/2, b.get_height()+0.1, f"{a*100:.2f}%", ha="center", fontweight="bold")
 for b, a in zip(bars, accuracies)]
bars2  = axes[1].bar(model_names, times, color=colors, edgecolor="black", width=0.5)
axes[1].set_title("Training Time (s)")
[axes[1].text(b.get_x()+b.get_width()/2, b.get_height()+0.1, f"{t:.1f}s", ha="center", fontweight="bold")
 for b, t in zip(bars2, times)]
plt.tight_layout(); save("D1_model_comparison.png")

# Confusion matrices
fig, axes = plt.subplots(1, 3, figsize=(21, 6))
fig.suptitle("Confusion Matrices", fontsize=14, fontweight="bold")
for ax, (name, info) in zip(axes, results.items()):
    ConfusionMatrixDisplay(info["cm"], display_labels=range(10)).plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"{name.replace(chr(10),' ')}  {info['acc']*100:.2f}%", fontsize=11, fontweight="bold")
plt.tight_layout(); save("D2_confusion_matrices.png")

# Loss curves
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Training Loss Curves", fontsize=14, fontweight="bold")
for ax, (mdl, lbl) in zip(axes, [(ann, "ANN"), (deep, "Deep MLP")]):
    ax.plot(mdl.loss_curve_, color="steelblue", linewidth=2)
    ax.set_title(f"{lbl} — Final Loss: {mdl.loss_:.4f}")
    ax.set_xlabel("Iteration"); ax.set_ylabel("Loss"); ax.grid(True, alpha=0.3)
plt.tight_layout(); save("D3_loss_curves.png")

# Best model predictions
best_name  = model_names[np.argmax(accuracies)]
best_pred  = results[best_name]["y_pred"]
np.random.seed(7); idx = np.random.choice(len(X_test), 20, replace=False)
fig = plt.figure(figsize=(20, 5))
fig.suptitle(f"Sample Predictions — {best_name.replace(chr(10),' ')}", fontsize=13, fontweight="bold")
for i, si in enumerate(idx):
    ax = fig.add_subplot(2, 10, i+1)
    ax.imshow(X_test[si].reshape(28, 28), cmap="gray")
    p, t = best_pred[si], y_test[si]
    ax.set_title(f"P:{p}\nT:{t}", fontsize=8, color="green" if p==t else "red", fontweight="bold")
    ax.axis("off")
plt.tight_layout(); save("E1_sample_predictions.png")

# Summary
print("\n" + "="*60)
df = pd.DataFrame({
    "Model":         [n.replace("\n"," ") for n in model_names],
    "Test Accuracy": [f"{a*100:.2f}%" for a in accuracies],
    "Train Time":    [f"{t:.1f}s" for t in times],
    "Error Rate":    [f"{(1-a)*100:.2f}%" for a in accuracies],
})
print(df.to_string(index=False))
print("="*60 + "\n")
