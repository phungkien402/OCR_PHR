"""Image preprocessing for OCR pipeline using OpenCV."""

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def preprocess_image(image_path: str, mode: str = "auto") -> np.ndarray:
    """Preprocess an image for OCR extraction.

    Args:
        image_path: Path to the input image.
        mode: Preprocessing mode - "lcd" for digital displays,
              "handwritten" for handwritten text, "auto" to detect.

    Returns:
        Preprocessed image as numpy array.
    """
    logger.info("Preprocessing image: %s", image_path)

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Resize if shortest dimension < 1000px (preserve aspect ratio)
    h, w = gray.shape[:2]
    min_dim = min(h, w)
    if min_dim < 1000:
        scale = 1000 / min_dim
        new_w = int(w * scale)
        new_h = int(h * scale)
        gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        logger.debug("Resized image from %dx%d to %dx%d", w, h, new_w, new_h)

    # Determine mode if auto
    if mode == "auto":
        mode = _detect_image_type(gray)
        logger.debug("Auto-detected image type: %s", mode)

    if mode == "lcd":
        processed = _preprocess_lcd(gray)
    else:
        processed = _preprocess_handwritten(gray)

    logger.info("Preprocessing complete (mode=%s)", mode)
    return processed


def preprocess_image_raw(image_path: str) -> np.ndarray:
    """Load image with minimal preprocessing (for EasyOCR which does its own).

    Args:
        image_path: Path to the input image.

    Returns:
        Image as BGR numpy array (resized if needed).
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    # Resize if too small
    h, w = img.shape[:2]
    min_dim = min(h, w)
    if min_dim < 800:
        scale = 800 / min_dim
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    return img


def _preprocess_lcd(gray: np.ndarray) -> np.ndarray:
    """Preprocess LCD/digital display image.

    Optimized for 7-segment displays with high contrast digits.
    """
    # Apply CLAHE with higher clip limit for LCD
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Light denoise (preserve sharp digit edges)
    denoised = cv2.fastNlMeansDenoising(enhanced, None, h=5, templateWindowSize=7, searchWindowSize=21)

    # Otsu thresholding for LCD/digital display images
    _, processed = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return processed


def _preprocess_handwritten(gray: np.ndarray) -> np.ndarray:
    """Preprocess handwritten text image.

    Optimized for varied strokes and textured backgrounds.
    """
    # Apply CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Denoise
    denoised = cv2.fastNlMeansDenoising(enhanced, None, h=10, templateWindowSize=7, searchWindowSize=21)

    # Adaptive thresholding for handwritten images
    processed = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )

    return processed


def _detect_image_type(image: np.ndarray) -> str:
    """Heuristic to detect if image is LCD display or handwritten.

    Uses edge density and contrast variance to distinguish between
    digital displays (sharp edges, uniform backgrounds) and
    handwritten text (varied strokes, textured backgrounds).
    """
    # Calculate edge density using Canny
    edges = cv2.Canny(image, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size

    # Calculate local variance
    local_var = cv2.Laplacian(image, cv2.CV_64F).var()

    # LCD displays tend to have lower edge density and higher local variance
    # due to sharp digit boundaries on uniform backgrounds
    if edge_density < 0.05 and local_var > 500:
        return "lcd"
    return "handwritten"
