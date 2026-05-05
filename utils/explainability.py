"""
Explainability utilities for MNIST CNN models
- Grad-CAM: Generate heatmaps showing which pixels drove predictions
- SHAP: Per-pixel attribution scores for model interpretability
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf
from tensorflow.keras.models import Model
import cv2
import warnings
warnings.filterwarnings("ignore")

def make_gradcam_heatmap(img_array, model, last_conv_layer_name="conv2d_2", pred_index=None):
    """
    Generate Grad-CAM heatmap for model prediction explanation.
    
    Args:
        img_array (np.ndarray): Preprocessed image array (1, 28, 28, 1)
        model (tf.keras.Model): Trained CNN model
        last_conv_layer_name (str): Name of last convolutional layer
        pred_index (int): Index of predicted class (None for top prediction)
        
    Returns:
        np.ndarray: Grad-CAM heatmap (28, 28)
    """
    # Create model that maps input to last conv layer outputs and predictions
    grad_model = Model(
        [model.inputs], 
        [model.get_layer(last_conv_layer_name).output, model.output]
    )
    
    # Compute gradient of top predicted class for input image
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    
    # Gradient of the predicted class with respect to output feature map
    grads = tape.gradient(class_channel, last_conv_layer_output)
    
    # Global average pooling of gradients to get importance weights
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Weighted combination of feature maps
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # Normalize heatmap to [0, 1]
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def overlay_gradcam(img, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    """
    Overlay Grad-CAM heatmap on original image.
    
    Args:
        img (np.ndarray): Original image (28, 28, 1)
        heatmap (np.ndarray): Grad-CAM heatmap (28, 28)
        alpha (float): Transparency factor for overlay
        colormap (int): OpenCV colormap for heatmap
        
    Returns:
        np.ndarray: Image with Grad-CAM overlay (28, 28, 3)
    """
    # Resize heatmap to match image size
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    
    # Convert heatmap to RGB
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, colormap)
    
    # Convert grayscale image to RGB
    if len(img.shape) == 2 or img.shape[2] == 1:
        img_rgb = cv2.cvtColor(img.squeeze(), cv2.COLOR_GRAY2RGB)
    else:
        img_rgb = img
    
    # Superimpose heatmap on original image
    superimposed_img = cv2.addWeighted(img_rgb, 1-alpha, heatmap, alpha, 0)
    
    return superimposed_img

def generate_gradcam_visualization(img_array, model, true_label=None, 
                                  last_conv_layer_name="conv2d_2", save_path=None):
    """
    Generate comprehensive Grad-CAM visualization with prediction and heatmap.
    
    Args:
        img_array (np.ndarray): Preprocessed image (1, 28, 28, 1)
        model (tf.keras.Model): Trained CNN model
        true_label (int): True digit label (optional)
        last_conv_layer_name (str): Name of last convolutional layer
        save_path (str): Path to save visualization (optional)
        
    Returns:
        tuple: (fig, predicted_digit, confidence)
    """
    # Get prediction
    preds = model.predict(img_array, verbose=0)[0]
    predicted_digit = np.argmax(preds)
    confidence = preds[predicted_digit] * 100
    
    # Generate Grad-CAM heatmap
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)
    
    # Convert back to original image format
    original_img = img_array[0].squeeze()
    
    # Create overlay
    overlay_img = overlay_gradcam(original_img, heatmap)
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(original_img, cmap='gray')
    axes[0].set_title(f'Original Image\n(True: {true_label if true_label is not None else "?"})', 
                     fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Grad-CAM heatmap
    im = axes[1].imshow(heatmap, cmap='jet')
    axes[1].set_title('Grad-CAM Heatmap\n(Pixel Importance)', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Overlay
    axes[2].imshow(overlay_img)
    axes[2].set_title(f'Prediction: {predicted_digit}\nConfidence: {confidence:.1f}%', 
                     fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Grad-CAM visualization saved: {save_path}")
    
    return fig, predicted_digit, confidence

def get_shap_values(model, X_background, X_test, max_samples=100):
    """
    Compute SHAP values for model explainability.
    
    Args:
        model (tf.keras.Model): Trained CNN model
        X_background (np.ndarray): Background samples for SHAP (n_samples, 28, 28, 1)
        X_test (np.ndarray): Test samples to explain (n_samples, 28, 28, 1)
        max_samples (int): Maximum number of test samples to explain
        
    Returns:
        shap.Explanation: SHAP explanation object
    """
    try:
        import shap
    except ImportError:
        raise ImportError("SHAP not installed. Install with: pip install shap")
    
    # Limit samples for computation efficiency
    if len(X_background) > 100:
        X_background = X_background[:100]
    if len(X_test) > max_samples:
        X_test = X_test[:max_samples]
    
    # Create SHAP explainer
    explainer = shap.DeepExplainer(model, X_background)
    
    # Compute SHAP values
    shap_values = explainer.shap_values(X_test)
    
    return shap_values, explainer

def plot_shap_explanation(img_array, shap_values, predicted_digit, save_path=None):
    """
    Visualize SHAP values for a single prediction.
    
    Args:
        img_array (np.ndarray): Input image (28, 28, 1)
        shap_values (np.ndarray): SHAP values for the image
        predicted_digit (int): Predicted digit class
        save_path (str): Path to save visualization (optional)
        
    Returns:
        plt.Figure: SHAP visualization figure
    """
    try:
        import shap
    except ImportError:
        raise ImportError("SHAP not installed. Install with: pip install shap")
    
    # Create SHAP plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Original image
    axes[0].imshow(img_array.squeeze(), cmap='gray')
    axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # SHAP values (pixel importance)
    shap_img = shap_values[predicted_digit].squeeze()
    im = axes[1].imshow(shap_img, cmap='RdBu_r', vmin=-shap_img.max(), vmax=shap_img.max())
    axes[1].set_title(f'SHAP Values\n(Pixel Attribution for Digit {predicted_digit})', 
                     fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    # Add colorbar
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"SHAP visualization saved: {save_path}")
    
    return fig

def create_explainability_report(model, X_test, y_test, num_samples=5, output_dir=None):
    """
    Generate comprehensive explainability report with Grad-CAM and SHAP.
    
    Args:
        model (tf.keras.Model): Trained CNN model
        X_test (np.ndarray): Test images (n_samples, 28, 28, 1)
        y_test (np.ndarray): Test labels (n_samples,)
        num_samples (int): Number of samples to analyze
        output_dir (str): Directory to save visualizations
        
    Returns:
        dict: Summary of explainability analysis
    """
    if output_dir is None:
        output_dir = "outputs/explainability"
    
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating explainability report...")
    
    # Select diverse samples (one from each digit class if possible)
    samples_per_class = min(num_samples // 10 + 1, len(X_test) // 10)
    selected_indices = []
    
    for digit in range(10):
        digit_indices = np.where(y_test == digit)[0]
        if len(digit_indices) > 0:
            selected = np.random.choice(digit_indices, 
                                     size=min(samples_per_class, len(digit_indices)), 
                                     replace=False)
            selected_indices.extend(selected)
    
    # Limit to requested number of samples
    selected_indices = selected_indices[:num_samples]
    
    results = {
        'grad_cam_samples': [],
        'shap_samples': [],
        'summary': {}
    }
    
    # Generate Grad-CAM visualizations
    print("  - Generating Grad-CAM heatmaps...")
    for i, idx in enumerate(selected_indices):
        img_array = X_test[idx:idx+1]
        true_label = y_test[idx]
        
        save_path = os.path.join(output_dir, f'gradcam_sample_{i}_digit_{true_label}.png')
        fig, pred_digit, confidence = generate_gradcam_visualization(
            img_array, model, true_label, save_path=save_path
        )
        plt.close(fig)
        
        results['grad_cam_samples'].append({
            'index': idx,
            'true_label': int(true_label),
            'predicted': int(pred_digit),
            'confidence': float(confidence),
            'correct': int(true_label) == int(pred_digit)
        })
    
    # Generate SHAP visualizations (computationally expensive, so fewer samples)
    print("  - Computing SHAP values...")
    try:
        # Use subset for SHAP background
        background_samples = X_test[:50]
        test_samples = X_test[selected_indices[:min(3, len(selected_indices))]]
        
        shap_values, explainer = get_shap_values(model, background_samples, test_samples)
        
        for i, idx in enumerate(selected_indices[:min(3, len(selected_indices))]):
            img_array = X_test[idx:idx+1]
            true_label = y_test[idx]
            
            # Get prediction for SHAP indexing
            preds = model.predict(img_array, verbose=0)[0]
            pred_digit = np.argmax(preds)
            
            save_path = os.path.join(output_dir, f'shap_sample_{i}_digit_{true_label}.png')
            fig = plot_shap_explanation(img_array, shap_values, pred_digit, save_path=save_path)
            plt.close(fig)
            
            results['shap_samples'].append({
                'index': idx,
                'true_label': int(true_label),
                'predicted': int(pred_digit),
                'shap_computed': True
            })
            
    except Exception as e:
        print(f"    SHAP computation failed: {str(e)}")
        print("    (This is normal if shap is not installed)")
    
    # Generate summary statistics
    grad_cam_correct = sum(1 for s in results['grad_cam_samples'] if s['correct'])
    results['summary'] = {
        'total_samples': len(results['grad_cam_samples']),
        'grad_cam_accuracy': grad_cam_correct / len(results['grad_cam_samples']) * 100,
        'shap_samples_generated': len(results['shap_samples']),
        'output_directory': output_dir
    }
    
    print(f"\n✓ Explainability report completed!")
    print(f"  - Grad-CAM visualizations: {len(results['grad_cam_samples'])}")
    print(f"  - SHAP visualizations: {len(results['shap_samples'])}")
    print(f"  - Accuracy on analyzed samples: {results['summary']['grad_cam_accuracy']:.1f}%")
    print(f"  - Saved to: {output_dir}")
    
    return results
