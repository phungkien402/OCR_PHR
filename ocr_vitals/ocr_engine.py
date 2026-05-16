"""OCR engine using Qwen3-VL:4b via Ollama for all image types.

Unified pipeline:
  1. Single Qwen3-VL:4b call with universal prompt (handles both LCD and handwritten)
  2. Fallback: VietOCR if Ollama is completely unavailable

No more LCD/handwritten branching or detect_image_mode().
"""

import logging
import os
import re

import cv2
import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

# Ollama configuration
OLLAMA_ENDPOINT = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "qwen3-vl:4b"

_vietocr_predictor = None


def load_vietocr(device: str = "cuda:0"):
    """Load VietOCR vgg_transformer model.

    Args:
        device: Target device (e.g. "cuda:0", "cuda:1", "cpu").

    Returns:
        VietOCR Predictor instance.
    """
    global _vietocr_predictor

    if _vietocr_predictor is not None:
        return _vietocr_predictor

    from vietocr.tool.config import Cfg
    from vietocr.tool.predictor import Predictor

    logger.info("Loading VietOCR vgg_transformer model on device: %s", device)

    config = Cfg.load_config_from_name("vgg_transformer")
    config["cnn"]["pretrained"] = True
    config["device"] = device

    try:
        _vietocr_predictor = Predictor(config)
        logger.info("VietOCR model loaded successfully on %s", device)
    except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
        if "cuda" in device:
            logger.warning(
                "Failed to load VietOCR on %s (%s). Falling back to CPU.", device, e
            )
            config["device"] = "cpu"
            _vietocr_predictor = Predictor(config)
            logger.info("VietOCR model loaded successfully on CPU (fallback)")
        else:
            raise

    return _vietocr_predictor


def extract_text(image: np.ndarray, device: str = "cuda:0", mode: str = "auto",
                  image_path: str = None) -> str:
    """Extract vital signs text from any medical image using Qwen3-VL:4b.

    Sends the image directly to Qwen3-VL via Ollama with a universal prompt
    that handles both LCD displays and handwritten/printed records.
    Falls back to VietOCR only if Ollama is completely unavailable.

    Args:
        image: Image as numpy array (grayscale or BGR).
        device: Target device (used only for VietOCR fallback).
        mode: Ignored — kept for API compatibility.
        image_path: Ignored — kept for API compatibility.

    Returns:
        Extracted text string.
    """
    text = _extract_text_qwen3_vl_universal(image)
    if text:
        logger.info("Qwen3-VL universal extraction successful")
        return text

    # Ollama unavailable — fall back to VietOCR
    logger.warning("Qwen3-VL unavailable, falling back to VietOCR")
    return _extract_text_vietocr(image, device)


def _extract_text_qwen3_vl_universal(image: np.ndarray) -> str:
    """Extract vital signs from any medical image using one universal prompt.

    Works for both:
    - LCD blood pressure monitors (reads SYS/DIA/PUL digits)
    - Handwritten or printed vitals records (reads all 7 vital sign fields)

    Args:
        image: BGR or grayscale image as numpy array.

    Returns:
        Extracted text string, or empty string if Ollama is unavailable.
    """
    import base64

    if len(image.shape) == 2:
        bgr_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        bgr_image = image

    success, img_encoded = cv2.imencode(".png", bgr_image)
    if not success:
        logger.error("Failed to encode image for Ollama")
        return ""

    img_b64 = base64.b64encode(img_encoded.tobytes()).decode()

    prompt = (
        "This is a medical vitals image. It may be a digital LCD blood pressure monitor "
        "or a handwritten/printed Vietnamese vitals record.\n\n"
        "Extract ALL vital sign values you can read. Look for:\n"
        "- Mạch / Pulse / Heart Rate / PUL (lần/phút)\n"
        "- Nhiệt độ / Temperature / TEMP (°C)\n"
        "- Huyết áp / Blood Pressure / SYS+DIA (mmHg)\n"
        "- Nhịp thở / Respiratory Rate / RR (lần/phút)\n"
        "- Cân nặng / Weight (kg)\n"
        "- Chiều cao / Height (cm)\n"
        "- SpO2 / Oxygen Saturation (%%)\n\n"
        "Return ONLY the values found in this exact format. "
        "Use null for any field not visible in the image:\n"
        "Mạch: <number or null>\n"
        "Nhiệt độ: <number or null>\n"
        "Huyết áp: <SYS>/<DIA> or null\n"
        "Nhịp thở: <number or null>\n"
        "Cân nặng: <number or null>\n"
        "Chiều cao: <number or null>\n"
        "SpO2: <number or null>"
    )

    return _call_ollama(img_b64, prompt)


def _call_ollama(img_b64: str, prompt: str) -> str:
    """Send a request to Ollama API with an image and prompt.

    Args:
        img_b64: Base64-encoded image string.
        prompt: Text prompt for the model.

    Returns:
        Model response text, or empty string on failure.
    """
    try:
        import requests
    except ImportError:
        logger.warning("requests library not available for Ollama API")
        return ""

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [{
            "role": "user",
            "content": prompt,
            "images": [img_b64]
        }],
        "stream": False
    }

    try:
        resp = requests.post(OLLAMA_ENDPOINT, json=payload, timeout=120)
        resp.raise_for_status()
        content = resp.json()["message"]["content"]
        logger.info("Qwen3-VL raw response: %s", repr(content[:200]))
        return content
    except requests.exceptions.ConnectionError:
        logger.warning("Ollama not running at %s", OLLAMA_ENDPOINT)
        return ""
    except requests.exceptions.Timeout:
        logger.warning("Ollama request timed out")
        return ""
    except Exception as e:
        logger.warning("Ollama request failed: %s", e)
        return ""


def _extract_text_vietocr(image: np.ndarray, device: str) -> str:
    """Extract text using VietOCR (optimized for Vietnamese handwritten text).

    Args:
        image: Image as numpy array.
        device: Target device.

    Returns:
        Extracted text string.
    """
    from .preprocessor import preprocess_for_handwritten

    predictor = load_vietocr(device)

    # Apply handwritten-specific thresholding
    if len(image.shape) == 2:
        binary = preprocess_for_handwritten(image)
        pil_image = Image.fromarray(binary).convert("RGB")
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        binary = preprocess_for_handwritten(gray)
        pil_image = Image.fromarray(binary).convert("RGB")

    # Detect text regions and OCR each one
    text_lines = _extract_regions_vietocr(pil_image, predictor)

    if not text_lines:
        # Fallback: OCR the entire image as one region
        logger.debug("No regions detected, running VietOCR on full image")
        text = predictor.predict(pil_image)
        return text.strip()

    return "\n".join(text_lines)


def _extract_regions_vietocr(pil_image: Image.Image, predictor) -> list:
    """Detect text regions using contour analysis and OCR each with VietOCR.

    Args:
        pil_image: PIL Image in RGB format.
        predictor: VietOCR predictor instance.

    Returns:
        List of extracted text strings, one per detected region.
    """
    # Convert to grayscale for contour detection
    img_array = np.array(pil_image.convert("L"))

    # Binary threshold
    _, binary = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Dilate to merge nearby text into regions
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 10))
    dilated = cv2.dilate(binary, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return []

    # Filter and sort contours by position (top to bottom, left to right)
    img_h, img_w = img_array.shape
    min_area = img_h * img_w * 0.001  # Minimum 0.1% of image area

    regions = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if area >= min_area and w > 20 and h > 10:
            regions.append((x, y, w, h))

    # Sort top-to-bottom, then left-to-right
    regions.sort(key=lambda r: (r[1] // 50, r[0]))

    text_lines = []
    for x, y, w, h in regions:
        # Add padding
        pad = 5
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(img_w, x + w + pad)
        y2 = min(img_h, y + h + pad)

        # Crop region from original RGB image
        region_img = pil_image.crop((x1, y1, x2, y2))

        try:
            text = predictor.predict(region_img)
            text = text.strip()
            if text:
                text_lines.append(text)
        except Exception as e:
            logger.debug("VietOCR failed on region (%d,%d,%d,%d): %s", x, y, w, h, e)

    return text_lines
