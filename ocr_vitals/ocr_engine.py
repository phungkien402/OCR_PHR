"""OCR engine using VietOCR vgg_transformer model."""

import logging

import numpy as np
import torch
from PIL import Image
from vietocr.tool.config import Cfg
from vietocr.tool.predictor import Predictor

logger = logging.getLogger(__name__)

_predictor = None


def load_model(device: str = "cuda:0") -> Predictor:
    """Load VietOCR vgg_transformer model.

    Args:
        device: Target device (e.g. "cuda:0", "cuda:1", "cpu").

    Returns:
        VietOCR Predictor instance.
    """
    global _predictor

    if _predictor is not None:
        return _predictor

    logger.info("Loading VietOCR vgg_transformer model on device: %s", device)

    config = Cfg.load_config_from_name("vgg_transformer")
    config["cnn"]["pretrained"] = True
    config["device"] = device

    try:
        _predictor = Predictor(config)
        logger.info("Model loaded successfully on %s", device)
    except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
        if "cuda" in device:
            logger.warning(
                "Failed to load model on %s (%s). Falling back to CPU.", device, e
            )
            config["device"] = "cpu"
            _predictor = Predictor(config)
            logger.info("Model loaded successfully on CPU (fallback)")
        else:
            raise

    return _predictor


def extract_text(image: np.ndarray, device: str = "cuda:0") -> str:
    """Extract text from a preprocessed image using VietOCR.

    Args:
        image: Preprocessed image as numpy array (grayscale or BGR).
        device: Target device for inference.

    Returns:
        Extracted text string.
    """
    predictor = load_model(device)

    # Convert numpy array to PIL Image
    if len(image.shape) == 2:
        # Grayscale -> RGB
        pil_image = Image.fromarray(image).convert("RGB")
    else:
        # BGR -> RGB
        pil_image = Image.fromarray(image[:, :, ::-1]).convert("RGB")

    # Detect text regions and OCR each one
    text_lines = _extract_regions(pil_image, predictor)

    if not text_lines:
        # Fallback: OCR the entire image as one region
        logger.debug("No regions detected, running OCR on full image")
        text = predictor.predict(pil_image)
        return text.strip()

    return "\n".join(text_lines)


def _extract_regions(pil_image: Image.Image, predictor: Predictor) -> list:
    """Detect text regions using contour analysis and OCR each region.

    Args:
        pil_image: PIL Image in RGB format.
        predictor: VietOCR predictor instance.

    Returns:
        List of extracted text strings, one per detected region.
    """
    import cv2

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
            logger.debug("OCR failed on region (%d,%d,%d,%d): %s", x, y, w, h, e)

    return text_lines
