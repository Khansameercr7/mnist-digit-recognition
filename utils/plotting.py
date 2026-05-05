"""
Plotting utilities for MNIST
- Saving plots with standard configuration
- Creating confusion matrices
- Plotting training metrics
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import numpy as np

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)


def save_plot(output_dir, filename, dpi=150):
    """
    Save current matplotlib figure to file.
    
    Args:
        output_dir (str): Directory to save to
        filename (str): Filename (e.g., 'confusion_matrix.png')
        dpi (int): Resolution in dots per inch
    """
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"  ✔  Saved: {filename}")


def plot_confusion_matrix(y_true, y_pred, output_dir, filename="confusion_matrix.png"):
    """
    Create and save confusion matrix plot.
    
    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
        output_dir (str): Output directory
        filename (str): Output filename
    """
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix")
    save_plot(output_dir, filename)


def plot_metrics(history, output_dir, metric="accuracy"):
    """
    Plot training and validation metrics over epochs.
    
    Args:
        history (dict or keras.History): Training history with keys like 'loss', 'val_loss', 'accuracy', 'val_accuracy'
        output_dir (str): Output directory
        metric (str): Metric to plot ('accuracy', 'loss', etc.)
    """
    # Handle Keras History object
    if hasattr(history, 'history'):
        history_dict = history.history
    else:
        history_dict = history
    
    plt.figure(figsize=(10, 6))
    
    if metric in history_dict:
        plt.plot(history_dict[metric], label=f"Training {metric}")
    if f"val_{metric}" in history_dict:
        plt.plot(history_dict[f"val_{metric}"], label=f"Validation {metric}")
    
    plt.title(f"{metric.capitalize()} over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel(metric.capitalize())
    plt.legend()
    plt.grid(True)
    
    save_plot(output_dir, f"{metric}_plot.png")


def plot_sample_predictions(X, y, predictions, output_dir, n_samples=10, filename="predictions_sample.png"):
    """
    Plot sample images with true and predicted labels.
    
    Args:
        X (np.ndarray): Input images (flattened or 2D)
        y (np.ndarray): True labels
        predictions (np.ndarray): Predicted labels or probabilities
        output_dir (str): Output directory
        n_samples (int): Number of samples to plot
        filename (str): Output filename
    """
    # Handle predictions as probabilities or labels
    if predictions.ndim > 1:
        pred_labels = np.argmax(predictions, axis=1)
    else:
        pred_labels = predictions
    
    n_samples = min(n_samples, len(X))
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    
    for i in range(n_samples):
        # Reshape if needed
        if X[i].ndim == 1:
            img = X[i].reshape(28, 28) if len(X[i]) == 784 else X[i].reshape(8, 8)
        else:
            img = X[i]
        
        axes[i].imshow(img, cmap="gray")
        axes[i].set_title(f"True: {y[i]}, Pred: {pred_labels[i]}")
        axes[i].axis("off")
    
    plt.tight_layout()
    save_plot(output_dir, filename)


def plot_class_distribution(y, output_dir, filename="class_distribution.png"):
    """
    Plot histogram of class distribution.
    
    Args:
        y (np.ndarray): Labels
        output_dir (str): Output directory
        filename (str): Output filename
    """
    plt.figure(figsize=(10, 6))
    plt.hist(y, bins=10, edgecolor="black", alpha=0.7)
    plt.xlabel("Digit Class")
    plt.ylabel("Frequency")
    plt.title("Class Distribution")
    plt.xticks(range(10))
    save_plot(output_dir, filename)
