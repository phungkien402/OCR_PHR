"""Image preprocessing for OCR pipeline using OpenCV."""

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def preprocess_image(image_path: str) -> np.ndarray:
    """Preprocess an image for OCR extraction.

    Returns a grayscale, contrast-enhanced, denoised image.
    Does NOT classify the image type — that is handled by ocr_engine.py.

    Args:
        image_path: Path to the input image.

    Returns:
        Preprocessed grayscale image as numpy array.
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

    # Apply CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Denoise
    denoised = cv2.fastNlMeansDenoising(enhanced, None, h=10, templateWindowSize=7, searchWindowSize=21)

    logger.info("Preprocessing complete")
    return denoised


def preprocess_for_handwritten(image: np.ndarray) -> np.ndarray:
    """Apply adaptive thresholding for handwritten text.

    Args:
        image: Grayscale preprocessed image.

    Returns:
        Binary image optimized for handwritten OCR.
    """
    processed = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    return processed


def preprocess_for_lcd(image: np.ndarray) -> np.ndarray:
    """Apply Otsu thresholding for LCD/digital display images.

    Args:
        image: Grayscale preprocessed image.

    Returns:
        Binary image optimized for LCD digit OCR.
    """
    _, processed = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return processed


def preprocess_image_raw(image_path: str) -> np.ndarray:
    """Load image with minimal preprocessing (for engines that do their own).

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
