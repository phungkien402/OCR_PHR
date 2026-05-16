"""OCR engine supporting Qwen3-VL (LCD displays) and VietOCR (handwritten text).

LCD pipeline:
  1. Screen detection + perspective warp (screen_detector)
  2. Qwen3-VL:2b via Ollama for digit recognition
  3. Fallback: Tesseract on warped image if Ollama unavailable

Image type detection is consolidated here — this is the single source of truth
for deciding which OCR engine to use.
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


def detect_image_mode(image: np.ndarray) -> str:
    """Detect whether image is LCD display or handwritten text.

    This is the SINGLE place for image type classification.
    Uses contrast ratio, histogram bimodality, and region analysis.

    Args:
        image: Grayscale or BGR image as numpy array.

    Returns:
        "lcd" or "handwritten"
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Calculate contrast ratio
    p5 = np.percentile(gray, 5)
    p95 = np.percentile(gray, 95)
    contrast_ratio = (p95 - p5) / 255.0

    # Check for dark background (LCD screens are typically dark)
    mean_val = np.mean(gray)
    dark_background = mean_val < 128

    # Calculate histogram bimodality (LCD displays tend to be bimodal)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist = hist / hist.sum()

    # Find peaks in histogram
    peaks = []
    for i in range(5, 251):
        if hist[i] > hist[i - 1] and hist[i] > hist[i + 1] and hist[i] > 0.01:
            peaks.append((i, hist[i]))

    # Analyze contours for large isolated digit-like regions
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    large_regions = 0
    if contours:
        img_area = gray.shape[0] * gray.shape[1]
        large_regions = sum(1 for c in contours if cv2.contourArea(c) > img_area * 0.003)

    # LCD detection heuristics:
    # 1. Dark background with some bright content (high contrast)
    # 2. OR product photo characteristics (medium contrast, few large regions)
    is_lcd = False

    # Strong signal: dark background + high contrast
    if dark_background and contrast_ratio > 0.3:
        is_lcd = True
    # Medium signal: high contrast + isolated large regions
    elif contrast_ratio > 0.4 and large_regions >= 2 and large_regions <= 40:
        is_lcd = True

    mode = "lcd" if is_lcd else "handwritten"
    logger.info(
        "Image type detection: %s (contrast=%.2f, large_regions=%d, peaks=%d, mean=%.0f)",
        mode, contrast_ratio, large_regions, len(peaks), mean_val,
    )
    return mode


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
    """Extract text from a preprocessed image.

    Args:
        image: Preprocessed image as numpy array (grayscale or BGR).
        device: Target device for inference.
        mode: OCR mode - "lcd", "handwritten", or "auto".
        image_path: Original image file path (needed for LCD warp pipeline).

    Returns:
        Extracted text string.
    """
    if mode == "auto":
        mode = detect_image_mode(image)

    if mode == "lcd":
        return _extract_text_lcd(image, image_path)
    else:
        return _extract_text_vietocr(image, device)


def _extract_text_lcd(image: np.ndarray, image_path: str = None) -> str:
    """Extract text from LCD display using Qwen3-VL via Ollama.

    Sends the original image directly to Qwen3-VL:4b without warping.
    The 4b model performs better on original images than warped ones.

    Fallback: Tesseract with screen warp if Ollama is unavailable.

    Args:
        image: Image as numpy array (grayscale or BGR).
        image_path: Path to the original image file.

    Returns:
        Extracted text string.
    """
    # Ensure BGR format for encoding
    if len(image.shape) == 2:
        bgr_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        bgr_image = image

    # Primary: Send original image directly to Qwen3-VL:4b
    text = _extract_text_qwen3_vl(bgr_image)
    if text:
        logger.info("Qwen3-VL extraction successful")
        return text

    # Fallback: Screen warp + Tesseract if Ollama unavailable
    logger.warning("Qwen3-VL unavailable, falling back to screen warp + Tesseract")

    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from screen_detector import detect_screen, warp_screen

    corners = detect_screen(bgr_image)
    if corners is not None:
        warped = warp_screen(bgr_image, corners)
        logger.info("Fallback: LCD screen warped for Tesseract")

        # Save debug warped image
        if image_path:
            os.makedirs("output", exist_ok=True)
            filename = os.path.basename(image_path)
            debug_path = f"output/debug_warped_{filename}"
            cv2.imwrite(debug_path, warped)

        return _extract_text_tesseract(warped)
    else:
        logger.warning("Screen detection also failed, using Tesseract on original")
        return _extract_text_tesseract(bgr_image)


def _extract_text_qwen3_vl(image: np.ndarray) -> str:
    """Send image to Qwen3-VL:2b via Ollama for LCD digit recognition.

    Args:
        image: BGR image as numpy array.

    Returns:
        Extracted text string, or empty string if Ollama is unavailable.
    """
    import base64

    try:
        import requests
    except ImportError:
        logger.warning("requests library not available for Ollama API")
        return ""

    # Encode image as base64 PNG
    success, img_encoded = cv2.imencode(".png", image)
    if not success:
        logger.error("Failed to encode image for Ollama")
        return ""

    img_b64 = base64.b64encode(img_encoded.tobytes()).decode()

    prompt = (
        "This is a blood pressure monitor. Look carefully at the LCD display. "
        "Read EXACTLY the numbers shown next to each label. "
        "SYS = systolic pressure (top number, usually 3 digits) "
        "DIA = diastolic pressure (middle number, usually 2 digits) "
        "PUL = pulse rate (bottom number, usually 2 digits) "
        "Reply ONLY in this exact format:\n"
        "SYS: <number>\n"
        "DIA: <number>\n"
        "PUL: <number>"
    )

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
        logger.info("Qwen3-VL raw response: %s", repr(content))
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


def _extract_text_tesseract(image: np.ndarray) -> str:
    """Extract text using Tesseract (optimized for LCD/7-segment displays).

    Runs multiple passes with different preprocessing to capture both
    labels (SYS, DIA, PUL) and 7-segment digits. Selects the best result.

    Args:
        image: Image as numpy array (grayscale or BGR).

    Returns:
        Extracted text string.
    """
    import pytesseract

    # Set tesseract path explicitly
    pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Strategy: try multiple threshold levels and PSM modes
    # LCD images have text (labels + digits) on dark backgrounds
    # A low binary threshold captures all text regardless of brightness

    results = []

    # Pass 1: Low threshold binary (captures all non-black content)
    _, binary_low = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    text = pytesseract.image_to_string(binary_low, lang="eng", config="--psm 11 --oem 3").strip()
    results.append(("binary_low_psm11", text))

    # Pass 2: Otsu threshold
    _, binary_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    text = pytesseract.image_to_string(binary_otsu, lang="eng", config="--psm 11 --oem 3").strip()
    results.append(("otsu_psm11", text))

    # Pass 3: Inverted with PSM 6
    inverted = cv2.bitwise_not(gray)
    text = pytesseract.image_to_string(inverted, lang="eng", config="--psm 6 --oem 3").strip()
    results.append(("inverted_psm6", text))

    # Pass 4: CLAHE enhanced with PSM 11
    clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(gray)
    text = pytesseract.image_to_string(enhanced, lang="eng", config="--psm 11 --oem 3").strip()
    results.append(("clahe_psm11", text))

    # Pass 5: Low threshold with PSM 6 (block of text)
    text = pytesseract.image_to_string(binary_low, lang="eng", config="--psm 6 --oem 3").strip()
    results.append(("binary_low_psm6", text))

    # Select best result: prefer one with LCD labels AND digits
    best_text = _select_best_tesseract_result(results)

    logger.info("Tesseract extracted %d chars", len(best_text))
    logger.debug("Tesseract output:\n%s", best_text)
    return best_text


def _select_best_tesseract_result(results: list) -> str:
    """Select the best Tesseract result from multiple passes.

    Prefers results that contain both LCD labels (SYS/DIA/PUL) and digit values.

    Args:
        results: List of (name, text) tuples.

    Returns:
        Best text result.
    """
    import re

    lcd_labels = {"sys", "dia", "pul"}

    scored = []
    for name, text in results:
        text_lower = text.lower()
        label_count = sum(1 for lbl in lcd_labels if lbl in text_lower)
        digit_sequences = re.findall(r"\d{2,3}", text)
        digit_count = len(digit_sequences)
        # Strongly prefer results with both labels and digits
        score = label_count * 3 + digit_count * 2
        if label_count > 0 and digit_count > 0:
            score += 10  # Bonus for having both
        scored.append((score, name, text))
        logger.debug("Tesseract pass '%s': score=%d, labels=%d, digits=%d",
                     name, score, label_count, digit_count)

    scored.sort(reverse=True)

    if scored and scored[0][0] > 0:
        logger.debug("Selected Tesseract result: %s (score=%d)", scored[0][1], scored[0][0])
        return scored[0][2]

    # Fallback: return longest non-empty result
    results_by_len = sorted(results, key=lambda r: len(r[1]), reverse=True)
    return results_by_len[0][1] if results_by_len else ""


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
