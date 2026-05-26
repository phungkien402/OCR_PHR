"""OCR engine using qwen2.5vl:7b via Ollama.

Optimizations:
- Image resized to max 1024px before base64 (biggest speed win for Ollama)
- num_predict=300 + temperature=0 → deterministic + fast
- <think> tokens stripped from Qwen3 output
- JSON output format → parser is trivial, no regex fragility
- Async-compatible: _qwen3_vl_extract_async for use with httpx in web_app
- VietOCR fallback unchanged
"""

import base64
import logging
import re

import cv2
import numpy as np

logger = logging.getLogger(__name__)

OLLAMA_ENDPOINT = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "qwen2.5vl:7b-q4_K_M"
MAX_IMAGE_DIM = 1024
OLLAMA_TIMEOUT = 45


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────

def extract_text(image: np.ndarray, device: str = "cuda:0",
                 mode: str = "auto", image_path: str = None) -> str:
    """Priority: Ollama → VietOCR fallback."""
    text = _qwen3_vl_extract(image)
    if text:
        return text

    logger.warning("Ollama unavailable — falling back to VietOCR")
    return _vietocr_extract(image, device)


async def extract_text_async(image: np.ndarray, device: str = "cuda:0") -> str:
    """Async version. Priority: Ollama → VietOCR."""
    text = await _qwen3_vl_extract_async(image)
    if text:
        return text

    logger.warning("Ollama unavailable — falling back to VietOCR")
    return _vietocr_extract(image, device)


# ─────────────────────────────────────────────
# Image helpers
# ─────────────────────────────────────────────

def _resize_for_vlm(image: np.ndarray, max_dim: int = MAX_IMAGE_DIM) -> np.ndarray:
    """Resize so the longest edge ≤ max_dim. Keeps aspect ratio."""
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= max_dim:
        return image
    scale = max_dim / longest
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _image_to_b64(image: np.ndarray) -> str:
    """Encode BGR/gray numpy image to base64 JPEG string."""
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise RuntimeError("Failed to encode image to JPEG")
    return base64.b64encode(buf.tobytes()).decode()


# ─────────────────────────────────────────────
# Prompt
# ─────────────────────────────────────────────

_VITAL_PROMPT = """You are a medical data entry assistant. Your job is to read EVERY label-value pair from the image exactly as written — do NOT skip any row, do NOT validate or filter values.

Label mapping (Vietnamese/English):
- Mạch / PUL / HR / Pulse / Heart Rate → mach
- Nhiệt độ / TEMP / Temperature → nhiet_do (°C, convert if °F)
- Huyết áp / SYS+DIA / Blood Pressure / BP → huyet_ap (format: systolic/diastolic)
- Nhịp thở / RR / Respiratory Rate → nhip_tho
- Cân nặng / Weight → can_nang (kg)
- Chiều cao / Height → chieu_cao (cm)
- SpO2 / SPO2 / O2 / Oxygen Sat → spo2

Rules:
- Report the EXACT number written next to each label, even if the value seems unusual
- For tables/spreadsheets: column A = label, column B or C = value — read each row
- Set null ONLY if the label is completely absent from the image

Return ONLY this JSON, no markdown, no explanation:
{"mach": int|null, "nhiet_do": float|null, "huyet_ap": {"tam_thu": int|null, "tam_truong": int|null}|null, "nhip_tho": int|null, "can_nang": float|null, "chieu_cao": float|null, "spo2": int|null}"""


# ─────────────────────────────────────────────
# Qwen2.5-VL extraction (sync)
# ─────────────────────────────────────────────

def _qwen3_vl_extract(image: np.ndarray) -> str:
    """Sync Ollama call. Returns raw model text or empty string on failure."""
    try:
        import requests
    except ImportError:
        return ""

    small = _resize_for_vlm(image)
    img_b64 = _image_to_b64(small)

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [{"role": "user", "content": _VITAL_PROMPT, "images": [img_b64]}],
        "stream": False,
        "options": {
            "temperature": 0,
            "num_predict": 300,
            "num_gpu": 0,
        },
    }

    try:
        resp = requests.post(OLLAMA_ENDPOINT, json=payload, timeout=OLLAMA_TIMEOUT)
        resp.raise_for_status()
        raw = resp.json()["message"]["content"]
        cleaned = _strip_think(raw)
        logger.debug("Qwen2.5-VL response: %s", cleaned[:200])
        return cleaned
    except requests.exceptions.ConnectionError:
        logger.warning("Ollama not running at %s", OLLAMA_ENDPOINT)
        return ""
    except requests.exceptions.Timeout:
        logger.warning("Ollama timed out after %ds", OLLAMA_TIMEOUT)
        return ""
    except Exception as e:
        logger.warning("Ollama error: %s", e)
        return ""


async def _qwen3_vl_extract_async(image: np.ndarray) -> str:
    """Async Ollama call using httpx — does not block the event loop."""
    try:
        import httpx
    except ImportError:
        logger.warning("httpx not installed — falling back to sync call")
        return _qwen3_vl_extract(image)

    small = _resize_for_vlm(image)
    img_b64 = _image_to_b64(small)

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [{"role": "user", "content": _VITAL_PROMPT, "images": [img_b64]}],
        "stream": False,
        "options": {"temperature": 0, "num_predict": 300},
    }

    try:
        async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as client:
            resp = await client.post(OLLAMA_ENDPOINT, json=payload)
            resp.raise_for_status()
            raw = resp.json()["message"]["content"]
            return _strip_think(raw)
    except httpx.ConnectError:
        logger.warning("Ollama not running at %s", OLLAMA_ENDPOINT)
        return ""
    except httpx.TimeoutException:
        logger.warning("Ollama async timed out after %ds", OLLAMA_TIMEOUT)
        return ""
    except Exception as e:
        logger.warning("Ollama async error: %s", e)
        return ""


def _strip_think(text: str) -> str:
    """Remove <think>...</think> blocks that Qwen3 reasoning model emits."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.strip()


# ─────────────────────────────────────────────
# VietOCR fallback
# ─────────────────────────────────────────────

def load_vietocr(device: str = "cuda:0"):
    global _vietocr_predictor
    if _vietocr_predictor is not None:
        return _vietocr_predictor

    import torch
    from vietocr.tool.config import Cfg
    from vietocr.tool.predictor import Predictor

    config = Cfg.load_config_from_name("vgg_transformer")
    config["cnn"]["pretrained"] = True
    config["device"] = device
    try:
        _vietocr_predictor = Predictor(config)
    except (RuntimeError, torch.cuda.OutOfMemoryError):
        config["device"] = "cpu"
        _vietocr_predictor = Predictor(config)
    return _vietocr_predictor


_vietocr_predictor = None


def _vietocr_extract(image: np.ndarray, device: str) -> str:
    from PIL import Image
    from .preprocessor import preprocess_for_handwritten

    predictor = load_vietocr(device)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    binary = preprocess_for_handwritten(gray)
    pil = Image.fromarray(binary).convert("RGB")

    lines = _vietocr_regions(pil, predictor)
    if lines:
        return "\n".join(lines)
    return predictor.predict(pil).strip()


def _vietocr_regions(pil_image, predictor) -> list:
    from PIL import Image
    img_arr = np.array(pil_image.convert("L"))
    _, binary = cv2.threshold(img_arr, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 10))
    dilated = cv2.dilate(binary, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    h, w = img_arr.shape
    min_area = h * w * 0.001
    regions = [(x, y, bw, bh) for c in contours
               for x, y, bw, bh in [cv2.boundingRect(c)]
               if bw * bh >= min_area and bw > 20 and bh > 10]
    regions.sort(key=lambda r: (r[1] // 50, r[0]))

    lines = []
    for x, y, bw, bh in regions:
        crop = pil_image.crop((max(0, x-5), max(0, y-5),
                               min(w, x+bw+5), min(h, y+bh+5)))
        try:
            t = predictor.predict(crop).strip()
            if t:
                lines.append(t)
        except Exception:
            pass
    return lines
