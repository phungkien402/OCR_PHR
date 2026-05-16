"""OCR engine supporting VietOCR (handwritten) and EasyOCR (LCD displays)."""

import logging

import cv2
import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

_vietocr_predictor = None
_easyocr_reader = None


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


def load_easyocr(device: str = "cuda:0"):
    """Load EasyOCR reader for English LCD digit recognition.

    Args:
        device: Target device.

    Returns:
        EasyOCR Reader instance.
    """
    global _easyocr_reader

    if _easyocr_reader is not None:
        return _easyocr_reader

    import easyocr

    use_gpu = "cuda" in device
    logger.info("Loading EasyOCR reader (gpu=%s)", use_gpu)

    try:
        _easyocr_reader = easyocr.Reader(["en"], gpu=use_gpu)
        logger.info("EasyOCR reader loaded successfully (gpu=%s)", use_gpu)
    except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
        if use_gpu:
            logger.warning(
                "Failed to load EasyOCR on GPU (%s). Falling back to CPU.", e
            )
            _easyocr_reader = easyocr.Reader(["en"], gpu=False)
            logger.info("EasyOCR reader loaded successfully on CPU (fallback)")
        else:
            raise

    return _easyocr_reader


def detect_image_mode(image: np.ndarray) -> str:
    """Auto-detect whether image is LCD display or handwritten text.

    Uses contrast ratio, edge characteristics, and region analysis
    to distinguish between digital displays and handwritten text.

    Args:
        image: Grayscale image as numpy array.

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

    # Calculate histogram bimodality (LCD displays tend to be bimodal)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist = hist / hist.sum()

    # Find peaks in histogram
    peaks = []
    for i in range(5, 251):
        if hist[i] > hist[i-1] and hist[i] > hist[i+1] and hist[i] > 0.01:
            peaks.append((i, hist[i]))

    # LCD displays: high contrast, bimodal histogram, fewer text regions
    # with large isolated digit-like contours
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        areas = [cv2.contourArea(c) for c in contours]
        img_area = gray.shape[0] * gray.shape[1]
        large_regions = [a for a in areas if a > img_area * 0.005]

        # LCD: few large isolated digit regions, high contrast
        if (contrast_ratio > 0.5 and len(large_regions) >= 2
                and len(large_regions) <= 30 and len(peaks) <= 4):
            logger.debug(
                "Detected LCD: contrast=%.2f, large_regions=%d, peaks=%d",
                contrast_ratio, len(large_regions), len(peaks)
            )
            return "lcd"

    logger.debug(
        "Detected handwritten: contrast=%.2f, peaks=%d",
        contrast_ratio, len(peaks)
    )
    return "handwritten"


def extract_text(image: np.ndarray, device: str = "cuda:0", mode: str = "auto") -> str:
    """Extract text from a preprocessed image.

    Args:
        image: Preprocessed image as numpy array (grayscale or BGR).
        device: Target device for inference.
        mode: OCR mode - "lcd", "handwritten", or "auto".

    Returns:
        Extracted text string.
    """
    if mode == "auto":
        mode = detect_image_mode(image)
        logger.info("Auto-detected OCR mode: %s", mode)

    if mode == "lcd":
        return _extract_text_easyocr(image, device)
    else:
        return _extract_text_vietocr(image, device)


def _extract_text_easyocr(image: np.ndarray, device: str) -> str:
    """Extract text using EasyOCR (optimized for LCD/7-segment displays).

    Uses spatial proximity to associate labels with their digit values,
    producing structured output like "SYS 128" even when they're on separate lines.

    Args:
        image: Image as numpy array.
        device: Target device.

    Returns:
        Extracted text string with labels and values associated.
    """
    reader = load_easyocr(device)

    # EasyOCR expects BGR or grayscale numpy array
    if len(image.shape) == 2:
        img_input = image
    else:
        img_input = image

    # Use lower thresholds to catch faint LCD digits
    results = reader.readtext(img_input, detail=1, paragraph=False,
                              text_threshold=0.3, low_text=0.3)

    if not results:
        return ""

    # Separate detections into labels and numbers
    labels = []  # (x_center, y_center, text)
    numbers = []  # (x_center, y_center, text)
    others = []  # (x_center, y_center, text)

    lcd_label_set = {"sys", "dia", "pul", "pulse"}

    for bbox, text, conf in results:
        x_center = (bbox[0][0] + bbox[2][0]) / 2
        y_center = (bbox[0][1] + bbox[2][1]) / 2
        text_clean = text.strip()

        if text_clean.lower().rstrip(":") in lcd_label_set:
            labels.append((x_center, y_center, text_clean))
        elif text_clean.replace(".", "").replace(",", "").isdigit():
            numbers.append((x_center, y_center, text_clean))
        else:
            others.append((x_center, y_center, text_clean))

    # Associate each label with the nearest number below or to the right
    text_lines = []
    used_numbers = set()

    for lx, ly, label_text in sorted(labels, key=lambda l: l[1]):
        best_num = None
        best_dist = float("inf")

        for i, (nx, ny, num_text) in enumerate(numbers):
            if i in used_numbers:
                continue

            # Number should be below the label (within 200px vertical)
            # or to the right on the same line (within 50px vertical, to the right)
            dy = ny - ly
            dx = nx - lx

            if 0 < dy < 200 and abs(dx) < 200:
                # Below the label
                dist = dy + abs(dx) * 0.5
            elif abs(dy) < 50 and dx > 0:
                # Same line, to the right
                dist = dx + abs(dy) * 2
            else:
                continue

            if dist < best_dist:
                best_dist = dist
                best_num = (i, num_text)

        if best_num is not None:
            used_numbers.add(best_num[0])
            text_lines.append(f"{label_text} {best_num[1]}")
        else:
            text_lines.append(label_text)

    # Add remaining numbers and other text
    for i, (nx, ny, num_text) in enumerate(numbers):
        if i not in used_numbers:
            text_lines.append(num_text)

    for ox, oy, other_text in others:
        text_lines.append(other_text)

    result = "\n".join(text_lines)
    logger.info("EasyOCR extracted %d lines, %d chars", len(text_lines), len(result))
    return result


def _extract_text_vietocr(image: np.ndarray, device: str) -> str:
    """Extract text using VietOCR (optimized for Vietnamese handwritten text).

    Args:
        image: Image as numpy array.
        device: Target device.

    Returns:
        Extracted text string.
    """
    predictor = load_vietocr(device)

    # Convert numpy array to PIL Image
    if len(image.shape) == 2:
        pil_image = Image.fromarray(image).convert("RGB")
    else:
        pil_image = Image.fromarray(image[:, :, ::-1]).convert("RGB")

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
