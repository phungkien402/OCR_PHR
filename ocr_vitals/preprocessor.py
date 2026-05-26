"""Image preprocessing for OCR pipeline.

Two separate paths:
- preprocess_for_vlm(): fast path for Qwen3-VL — just resize, no denoising
- preprocess_image(): slow path for VietOCR — CLAHE + denoising for handwritten text
"""

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def preprocess_for_vlm(image_path: str, max_dim: int = 1024) -> np.ndarray:
    """Fast preprocessing for vision-language models (Qwen3-VL).

    VLMs do their own internal preprocessing — sending CLAHE/denoised grayscale
    actually HURTS accuracy (model expects natural color images).
    Just resize to keep inference fast; return BGR.

    Args:
        image_path: Path to input image.
        max_dim: Max longest edge in pixels (default 1024).

    Returns:
        BGR image resized to max_dim.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    h, w = img.shape[:2]
    longest = max(h, w)
    if longest > max_dim:
        scale = max_dim / longest
        img = cv2.resize(img, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_AREA)
        logger.debug("VLM resize: %dx%d → %dx%d", w, h, img.shape[1], img.shape[0])

    return img


def preprocess_image(image_path: str) -> np.ndarray:
    """Full preprocessing for VietOCR (handwritten text).

    Grayscale + CLAHE contrast enhancement + denoising.
    Slow but improves VietOCR accuracy on low-contrast scanned text.

    Args:
        image_path: Path to input image.

    Returns:
        Preprocessed grayscale image.
    """
    logger.info("Preprocessing for VietOCR: %s", image_path)
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Upscale small images for better OCR
    h, w = gray.shape[:2]
    if min(h, w) < 1000:
        scale = 1000 / min(h, w)
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)),
                          interpolation=cv2.INTER_CUBIC)

    # CLAHE contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Denoising (slow but helps VietOCR on noisy scans)
    denoised = cv2.fastNlMeansDenoising(
        enhanced, None, h=10, templateWindowSize=7, searchWindowSize=21
    )
    return denoised


def preprocess_image_raw(image_path: str) -> np.ndarray:
    """Load image for VietOCR with minimal preprocessing."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")
    h, w = img.shape[:2]
    if min(h, w) < 800:
        scale = 800 / min(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_CUBIC)
    return img


def preprocess_for_handwritten(image: np.ndarray) -> np.ndarray:
    """Adaptive threshold for handwritten text (VietOCR input)."""
    return cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )


def preprocess_for_lcd(image: np.ndarray) -> np.ndarray:
    """Otsu threshold for LCD displays."""
    _, out = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return out
