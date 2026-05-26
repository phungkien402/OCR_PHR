"""Main entry point for OCR vital signs extraction pipeline.

Adds process_image_async() for non-blocking use in FastAPI endpoints.
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

from .config import DEVICE
from .ocr_engine import extract_text, extract_text_async, OLLAMA_MODEL
from .parser import parse_vitals
from .preprocessor import preprocess_for_vlm, preprocess_image_raw
from .validator import validate_vitals

logger = logging.getLogger(__name__)
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


# ─────────────────────────────────────────────
# Core pipeline (sync — for CLI)
# ─────────────────────────────────────────────

def process_image(image_path: str, device: str = "cuda:0", mode: str = "auto") -> dict:
    """Sync pipeline — for CLI usage."""
    filename = os.path.basename(image_path)
    try:
        raw_img = preprocess_for_vlm(image_path)
        raw_text = extract_text(raw_img, device=device)
        return _build_result(filename, raw_text)
    except Exception as e:
        logger.error("Error processing %s: %s", filename, e)
        return _error_result(filename, str(e))


# ─────────────────────────────────────────────
# Async pipeline (for FastAPI)
# ─────────────────────────────────────────────

async def process_image_async(image_path: str, device: str = "cuda:0") -> dict:
    """Async pipeline — use in FastAPI endpoints to avoid blocking.

    Uses httpx for the Ollama call so the event loop stays free.
    """
    filename = os.path.basename(image_path)
    try:
        raw_img = preprocess_for_vlm(image_path)
        raw_text = await extract_text_async(raw_img, device=device)
        return _build_result(filename, raw_text)
    except Exception as e:
        logger.error("Error processing %s: %s", filename, e)
        return _error_result(filename, str(e))


# ─────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────

def _build_result(filename: str, raw_text: str) -> dict:
    """Parse raw OCR text into a full result dict."""
    from .config import VITALS_INFO

    vitals = parse_vitals(raw_text)
    units = vitals.pop("_units", None)
    validation, missing_fields = validate_vitals(vitals)

    fields_meta = {
        field: {
            "label_vn": info["label_vn"],
            "label_en": info["label_en"],
            "unit": info["unit"],
            "normal_range": info["normal_range"],
            "value": vitals.get(field),
        }
        for field, info in VITALS_INFO.items()
    }

    result = {
        "source_image": filename,
        "ocr_raw_text": raw_text,
        "vitals": vitals,
        "fields_meta": fields_meta,
        "validation": validation,
        "missing_fields": missing_fields,
        "ocr_engine": _detect_engine(),
        "processed_at": datetime.now().isoformat(timespec="seconds"),
    }
    if units:
        result["units_detected"] = units
    return result


def _detect_engine() -> str:
    return f"ollama/{OLLAMA_MODEL}"


def _error_result(filename: str, error: str) -> dict:
    return {
        "source_image": filename,
        "error": error,
        "ocr_engine": "unknown",
        "processed_at": datetime.now().isoformat(timespec="seconds"),
    }


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def save_result(result: dict, output_dir: str):
    source = result.get("source_image", "unknown")
    out_path = os.path.join(output_dir, f"{Path(source).stem}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    logger.info("Saved: %s", out_path)


def get_image_files(input_path: str) -> list:
    input_path = os.path.abspath(input_path)
    if os.path.isfile(input_path):
        return [input_path] if Path(input_path).suffix.lower() in SUPPORTED_EXTENSIONS else []
    if os.path.isdir(input_path):
        return sorted(
            os.path.join(input_path, f) for f in os.listdir(input_path)
            if Path(f).suffix.lower() in SUPPORTED_EXTENSIONS
        )
    return []


def main():
    parser = argparse.ArgumentParser(description="Extract vital signs from medical images")
    parser.add_argument("--input", "-i", required=True)
    parser.add_argument("--output", "-o", required=True)
    parser.add_argument("--device", "-d", default=DEVICE)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    os.makedirs(args.output, exist_ok=True)
    files = get_image_files(args.input)
    if not files:
        logger.warning("No images found")
        sys.exit(0)

    ok = err = 0
    for path in files:
        result = process_image(path, device=args.device)
        save_result(result, args.output)
        if "error" in result:
            err += 1
        else:
            ok += 1
    logger.info("Done. OK=%d  Error=%d", ok, err)


if __name__ == "__main__":
    main()
