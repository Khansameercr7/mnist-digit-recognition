"""
Preprocessing utilities for MNIST
- Image preprocessing (base64, PIL, numpy conversions)
- Data augmentation (rotation, zoom, shift)
"""

import io
import base64
import numpy as np
from PIL import Image, ImageFilter
from scipy.ndimage import zoom, rotate, shift as nd_shift


def preprocess_image(image_data, use_mnist_preprocessing=True):
    """
    Convert base64/PIL image → model-ready (1, 28, 28, 1) numpy array.
    
    Args:
        image_data (str or PIL.Image): Base64 encoded image or PIL Image object
        use_mnist_preprocessing (bool): Apply MNIST-style preprocessing (recommended for drawn digits)
        
    Returns:
        np.ndarray: Preprocessed image array of shape (1, 28, 28, 1)
    """
    if isinstance(image_data, str):
        # Strip data URI prefix if present
        if "," in image_data:
            image_data = image_data.split(",")[1]
        img_bytes = base64.b64decode(image_data)
        img = Image.open(io.BytesIO(img_bytes))
    else:
        img = image_data

    # Apply MNIST preprocessing for all inputs
    arr = preprocess_canvas_image(img)
    return arr

def preprocess_canvas_image(img):
    """
    Simplified preprocessing for hand-drawn digits.
    
    Pipeline (optimized for canvas input):
    1. Convert to grayscale
    2. Slight blur (reduce drawing artifacts)
    3. Simple threshold (binary)
    4. Invert if needed (white digit on black)
    5. Resize to 28x28
    6. Normalize (0-1)
    
    Args:
        img (PIL.Image): Input image
        
    Returns:
        np.ndarray: Preprocessed image array of shape (1, 28, 28, 1)
    """
    try:
        import cv2
    except ImportError:
        raise ImportError("OpenCV not installed. Install with: pip install opencv-python")
    
    # 1. Convert to grayscale
    img = img.convert('L')
    img_array = np.array(img)
    
    # 2. Light Gaussian blur to reduce noise from pen strokes
    img_array = cv2.GaussianBlur(img_array, (3, 3), 0)
    
    # 3. Simple threshold (convert to binary)
    # For hand-drawn: darker pixels = digit strokes
    _, binary = cv2.threshold(img_array, 100, 255, cv2.THRESH_BINARY)
    
    # 4. Invert colors if needed
    # Count white and black pixels
    white_pixels = np.sum(binary == 255)
    black_pixels = np.sum(binary == 0)
    
    # If mostly white with black digit, invert to white digit on black
    if white_pixels > black_pixels:
        binary = 255 - binary
    
    # 5. Resize to 28x28 (direct resize, preserve aspect)
    digit_28 = cv2.resize(binary, (28, 28), interpolation=cv2.INTER_AREA)
    
    # 6. Normalize to 0-1 range
    digit_28 = digit_28.astype("float32") / 255.0
    
    # Reshape for CNN input
    digit_28 = digit_28.reshape(1, 28, 28, 1)
    
    return digit_28

def preprocess_for_mnist(img):
    """
    Apply MNIST-style preprocessing to match training data distribution.
    
    This addresses the key issues with drawn digits:
    1. Centering and cropping the digit
    2. Proper threshold and inversion
    3. Stroke thickness normalization
    
    Args:
        img (PIL.Image): Input grayscale image
        
    Returns:
        PIL.Image: Preprocessed image (28x28)
    """
    # Convert to numpy array
    img_array = np.array(img)
    
    # Step 1: Threshold to binary (black and white)
    # MNIST has white digits on black background
    threshold = 128
    binary = (img_array < threshold).astype(np.uint8) * 255
    
    # Step 2: Find bounding box of the digit
    if np.sum(binary) > 0:  # Only if there's content
        # Find non-zero pixels
        rows = np.any(binary, axis=1)
        cols = np.any(binary, axis=0)
        
        # Get bounding box
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        
        # Add padding
        padding = 4
        y_min = max(0, y_min - padding)
        y_max = min(binary.shape[0], y_max + padding)
        x_min = max(0, x_min - padding)
        x_max = min(binary.shape[1], x_max + padding)
        
        # Crop to bounding box
        cropped = binary[y_min:y_max, x_min:x_max]
    else:
        # If no content found, use original
        cropped = binary
    
    # Step 3: Resize to square while maintaining aspect ratio
    if cropped.shape[0] == 0 or cropped.shape[1] == 0:
        # Fallback to original
        return Image.fromarray(img_array).resize((28, 28), Image.LANCZOS)
    
    # Get dimensions
    h, w = cropped.shape
    
    # Calculate scaling to fit in 20x20 (leaving room for padding)
    scale = min(20 / h, 20 / w)
    new_h, new_w = int(h * scale), int(w * scale)
    
    # Resize
    from PIL import Image
    cropped_img = Image.fromarray(cropped)
    resized = cropped_img.resize((new_w, new_h), Image.LANCZOS)
    
    # Step 4: Center in 28x28 canvas
    centered = Image.new('L', (28, 28), 0)  # Black background
    offset_x = (28 - new_w) // 2
    offset_y = (28 - new_h) // 2
    centered.paste(resized, (offset_x, offset_y))
    
    # Step 5: Apply slight blur to match MNIST distribution
    # MNIST digits aren't perfectly sharp
    centered = centered.filter(ImageFilter.GaussianBlur(radius=0.5))
    
    return centered


def augment_image(image, rotation_range=15, zoom_range=(0.85, 1.15), shift_range=2):
    """
    Apply random augmentation to a single image.
    
    Args:
        image (np.ndarray): Input image array (8x8 or 28x28)
        rotation_range (float): Max rotation in degrees
        zoom_range (tuple): (min_zoom, max_zoom) factors
        shift_range (int): Max shift in pixels
        
    Returns:
        np.ndarray: Augmented image
    """
    # Reshape to 2D if flattened
    is_flat = image.ndim == 1
    size = int(np.sqrt(len(image))) if is_flat else image.shape[0]
    img = image.reshape(size, size) if is_flat else image
    
    # Random rotation
    angle = np.random.uniform(-rotation_range, rotation_range)
    img = rotate(img, angle, reshape=False, order=1)
    
    # Random zoom
    zoom_factor = np.random.uniform(zoom_range[0], zoom_range[1])
    img = zoom(img, zoom_factor, order=1)
    if img.shape != (size, size):
        center = img.shape[0] // 2
        img = img[center - size//2:center + size//2, 
                  center - size//2:center + size//2]
    
    # Random shift
    shift_x = np.random.randint(-shift_range, shift_range + 1)
    shift_y = np.random.randint(-shift_range, shift_range + 1)
    img = nd_shift(img, (shift_x, shift_y), order=1)
    
    return img.flatten() if is_flat else img


def augment_dataset(X, y, target_size, augmentation_params=None):
    """
    Augment dataset to reach target size.
    
    Args:
        X (np.ndarray): Input features
        y (np.ndarray): Labels
        target_size (int): Target number of samples
        augmentation_params (dict): Augmentation parameters
        
    Returns:
        tuple: (augmented_X, augmented_y)
    """
    if augmentation_params is None:
        augmentation_params = {}
    
    X_aug = [X.copy()]
    y_aug = [y.copy()]
    
    current_size = len(X)
    while current_size < target_size:
        idx = np.random.randint(0, len(X))
        aug_img = augment_image(X[idx], **augmentation_params)
        X_aug.append(aug_img)
        y_aug.append(y[idx])
        current_size += 1
    
    return np.array(X_aug), np.array(y_aug)


def normalize_image(image, mean=0.0, std=1.0):
    """
    Normalize image to mean and std.
    
    Args:
        image (np.ndarray): Input image
        mean (float): Target mean
        std (float): Target standard deviation
        
    Returns:
        np.ndarray: Normalized image
    """
    return (image - image.mean()) / (image.std() + 1e-8) * std + mean
