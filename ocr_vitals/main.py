"""Main entry point for OCR vital signs extraction pipeline."""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

from .config import DEVICE
from .ocr_engine import extract_text, detect_image_mode
from .parser import parse_vitals
from .preprocessor import preprocess_image, preprocess_image_raw
from .validator import validate_vitals

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def process_image(image_path: str, device: str, mode: str = "auto") -> dict:
    """Process a single image and extract vital signs.

    Args:
        image_path: Path to the input image.
        device: Device for OCR inference.
        mode: OCR mode - "lcd", "handwritten", or "auto".

    Returns:
        Result dictionary with extracted vitals and metadata.
    """
    filename = os.path.basename(image_path)
    logger.info("Processing image: %s (mode=%s)", filename, mode)

    try:
        # Step 1: Load raw image for detection and LCD OCR
        raw_img = preprocess_image_raw(image_path)

        # Step 2: Determine OCR mode
        if mode == "auto":
            detected_mode = detect_image_mode(raw_img)
        else:
            detected_mode = mode

        # Step 3: OCR based on detected mode
        if detected_mode == "lcd":
            # Qwen3-VL via Ollama with screen warp preprocessing
            raw_text = extract_text(raw_img, device=device, mode="lcd",
                                    image_path=image_path)
            ocr_engine_used = "qwen3_vl_2b_ollama"
        else:
            # VietOCR needs preprocessed image
            preprocessed = preprocess_image(image_path)
            raw_text = extract_text(preprocessed, device=device, mode="handwritten")
            ocr_engine_used = "vietocr_vgg_transformer"

        logger.info("OCR extracted text length: %d chars", len(raw_text))

        # Step 3: Parse vitals
        vitals = parse_vitals(raw_text)

        # Remove internal metadata before output
        units = vitals.pop("_units", None)

        # Step 4: Validate
        validation, missing_fields = validate_vitals(vitals)

        result = {
            "source_image": filename,
            "ocr_raw_text": raw_text,
            "vitals": vitals,
            "validation": validation,
            "missing_fields": missing_fields,
            "ocr_engine": ocr_engine_used,
            "processed_at": datetime.now().isoformat(timespec="seconds"),
        }

        if units:
            result["units_detected"] = units

    except Exception as e:
        logger.error("Error processing %s: %s", filename, e)
        result = {
            "source_image": filename,
            "error": str(e),
            "ocr_engine": "unknown",
            "processed_at": datetime.now().isoformat(timespec="seconds"),
        }

    return result


def save_result(result: dict, output_dir: str):
    """Save result dictionary as JSON file.

    Args:
        result: Result dictionary to save.
        output_dir: Output directory path.
    """
    source = result.get("source_image", "unknown")
    json_filename = f"{Path(source).stem}.json"
    output_path = os.path.join(output_dir, json_filename)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    logger.info("Saved result to: %s", output_path)


def get_image_files(input_path: str) -> list:
    """Get list of image files from input path.

    Args:
        input_path: Path to a single image or directory.

    Returns:
        List of image file paths.
    """
    input_path = os.path.abspath(input_path)

    if os.path.isfile(input_path):
        ext = Path(input_path).suffix.lower()
        if ext in SUPPORTED_EXTENSIONS:
            return [input_path]
        else:
            logger.warning("Unsupported file extension: %s", ext)
            return []

    elif os.path.isdir(input_path):
        files = []
        for entry in sorted(os.listdir(input_path)):
            filepath = os.path.join(input_path, entry)
            if os.path.isfile(filepath):
                ext = Path(filepath).suffix.lower()
                if ext in SUPPORTED_EXTENSIONS:
                    files.append(filepath)
        logger.info("Found %d image(s) in directory: %s", len(files), input_path)
        return files

    else:
        logger.error("Input path does not exist: %s", input_path)
        return []


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Extract vital signs from medical images using OCR"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to input image or directory of images",
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output directory for JSON results",
    )
    parser.add_argument(
        "--device", "-d",
        default=DEVICE,
        help=f"Device for OCR inference (default: {DEVICE})",
    )
    parser.add_argument(
        "--mode", "-m",
        choices=["auto", "lcd", "handwritten"],
        default="auto",
        help="OCR mode: lcd (EasyOCR for digital displays), "
             "handwritten (VietOCR for Vietnamese text), "
             "auto (detect automatically, default)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose (DEBUG) logging",
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Ensure output directory exists
    os.makedirs(args.output, exist_ok=True)

    # Get image files
    image_files = get_image_files(args.input)
    if not image_files:
        logger.warning("No images found to process")
        sys.exit(0)

    logger.info("Processing %d image(s) on device: %s (mode: %s)", len(image_files), args.device, args.mode)

    # Process each image
    success_count = 0
    error_count = 0

    for image_path in image_files:
        result = process_image(image_path, device=args.device, mode=args.mode)
        save_result(result, args.output)

        if "error" in result:
            error_count += 1
        else:
            success_count += 1

    logger.info(
        "Done. Processed: %d, Errors: %d, Total: %d",
        success_count, error_count, len(image_files),
    )


if __name__ == "__main__":
    main()
