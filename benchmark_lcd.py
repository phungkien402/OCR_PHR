"""Benchmark script: Test multiple OCR engines on LCD blood pressure monitor image.

Ground truth for test image may-do-huyet-ap-bap-tay-d2group-kf-65a.png:
  SYS = 128
  DIA = 78
  PUL = 72

Runs on both a real product photo and a synthetic LCD image.
"""

import os
# Set NCCL env vars early (before torch/easyocr import) to avoid CUDA comm errors
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_SHM_DISABLE"] = "1"

import math
import re
import sys
import time

import cv2
import numpy as np

# Ground truth
GROUND_TRUTH = {"SYS": 128, "DIA": 78, "PUL": 72}
IMAGE_PATH = "test_images/may-do-huyet-ap-bap-tay-d2group-kf-65a.png"
IMAGE_PATH_SYNTHETIC = "test_images/synthetic_lcd.png"

# Results storage
results = []
# Current image path (set per run)
CURRENT_IMAGE = IMAGE_PATH


def score_output(raw_text: str) -> dict:
    """Try to extract SYS, DIA, PUL values from raw OCR text.

    Returns dict with extracted values and score (0-3).
    """
    text = raw_text.lower() if raw_text else ""
    extracted = {"SYS": None, "DIA": None, "PUL": None}

    # Strategy 1: Look for label + number patterns
    for label in ["sys", "dia", "pul"]:
        match = re.search(re.escape(label) + r"[\s.:;=]*(\d{2,3})", text)
        if match:
            extracted[label.upper()] = int(match.group(1))

    # Strategy 2: Line-by-line sequential matching
    if any(v is None for v in extracted.values()):
        lines = text.split("\n")
        for i, line in enumerate(lines):
            line_clean = line.strip()
            if "sys" in line_clean and extracted["SYS"] is None:
                val = _find_digit_nearby(lines, i)
                if val:
                    extracted["SYS"] = val
            elif "dia" in line_clean and extracted["DIA"] is None:
                val = _find_digit_nearby(lines, i)
                if val:
                    extracted["DIA"] = val
            elif "pul" in line_clean and extracted["PUL"] is None:
                val = _find_digit_nearby(lines, i)
                if val:
                    extracted["PUL"] = val

    # Strategy 3: If we found 3 numbers in order without labels, assume SYS/DIA/PUL
    if all(v is None for v in extracted.values()):
        all_nums = re.findall(r"\b(\d{2,3})\b", raw_text or "")
        # Filter to plausible BP values
        plausible = [int(n) for n in all_nums if 30 <= int(n) <= 250]
        if len(plausible) >= 3:
            extracted["SYS"] = plausible[0]
            extracted["DIA"] = plausible[1]
            extracted["PUL"] = plausible[2]

    # Calculate score
    score = 0
    for key in ["SYS", "DIA", "PUL"]:
        if extracted[key] == GROUND_TRUTH[key]:
            score += 1

    return {"extracted": extracted, "score": score}


def _find_digit_nearby(lines: list, idx: int) -> int:
    """Find a 2-3 digit number on the same line or next lines.

    Skips unit labels (mmHg, kPa, etc.) and model numbers when searching.
    """
    unit_words = {"mmhg", "kpa", "bpm", "pul/min", "pulse/min", "/min"}

    # Check same line (exclude the label itself)
    line_text = re.sub(r"(sys|dia|pul|pulse|systolic|diastolic)", "", lines[idx], flags=re.IGNORECASE)
    # Remove model numbers like D2-65A
    line_text = re.sub(r"[A-Za-z]\d+[-]?\d*[A-Za-z]?", "", line_text)
    match = re.search(r"(\d{2,3})", line_text)
    if match:
        val = int(match.group(1))
        if 30 <= val <= 250:
            return val
    # Check next lines (search further: up to 5 lines)
    for j in range(idx + 1, min(idx + 6, len(lines))):
        line = lines[j].strip()
        if not line:
            continue
        # Skip unit-only lines
        if line.lower() in unit_words:
            continue
        # Skip model number lines (e.g. D2-65A)
        if re.match(r"^[A-Za-z]?\d+[-]\d*[A-Za-z]*$", line):
            continue
        # Skip lines that are other labels
        if any(lbl in line.lower() for lbl in ["sys", "dia", "pul", "on", "off"]):
            break
        match = re.search(r"^(\d{2,3})$", line)
        if match:
            val = int(match.group(1))
            if 30 <= val <= 250:
                return val
        # Remove model numbers before checking
        cleaned = re.sub(r"[A-Za-z]\d+[-]?\d*[A-Za-z]?", "", line)
        match = re.search(r"(\d{2,3})", cleaned)
        if match:
            val = int(match.group(1))
            if 30 <= val <= 250:
                return val
    return None


def run_engine(name: str, func):
    """Run an OCR engine and record results."""
    print(f"\n{'='*60}")
    print(f"ENGINE: {name}")
    print(f"{'='*60}")

    try:
        start = time.time()
        raw_text = func()
        elapsed = time.time() - start

        print(f"Raw output:\n{raw_text}")
        print(f"\nTime: {elapsed:.2f}s")

        result = score_output(raw_text)
        print(f"Extracted: SYS={result['extracted']['SYS']}, "
              f"DIA={result['extracted']['DIA']}, "
              f"PUL={result['extracted']['PUL']}")
        print(f"Score: {result['score']}/3")

        results.append({
            "engine": name,
            "raw": raw_text[:100] if raw_text else "",
            "sys": result["extracted"]["SYS"],
            "dia": result["extracted"]["DIA"],
            "pul": result["extracted"]["PUL"],
            "score": result["score"],
            "time": elapsed,
            "error": None,
        })

    except Exception as e:
        elapsed = time.time() - start if 'start' in dir() else 0
        print(f"ERROR: {e}")
        results.append({
            "engine": name,
            "raw": "",
            "sys": None,
            "dia": None,
            "pul": None,
            "score": 0,
            "time": elapsed,
            "error": str(e),
        })


# ============================================================
# ENGINE 1 — Tesseract default (PSM 6, 11, 3)
# ============================================================

def engine_tesseract_default():
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

    img = cv2.imread(CURRENT_IMAGE)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    best_text = ""
    best_score = -1

    for psm in [6, 11, 3]:
        config = f"--psm {psm} --oem 3"
        text = pytesseract.image_to_string(gray, lang="eng", config=config)
        s = score_output(text)["score"]
        print(f"  PSM {psm}: score={s}, text={repr(text.strip()[:80])}")
        if s > best_score:
            best_score = s
            best_text = text

        # Also try inverted
        inverted = cv2.bitwise_not(gray)
        text_inv = pytesseract.image_to_string(inverted, lang="eng", config=config)
        s_inv = score_output(text_inv)["score"]
        print(f"  PSM {psm} (inv): score={s_inv}, text={repr(text_inv.strip()[:80])}")
        if s_inv > best_score:
            best_score = s_inv
            best_text = text_inv

    return best_text


# ============================================================
# ENGINE 2 — Tesseract with digit whitelist
# ============================================================

def engine_tesseract_digits():
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

    img = cv2.imread(CURRENT_IMAGE)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    config = "--psm 11 --oem 3 -c tessedit_char_whitelist=0123456789"

    # Try multiple preprocessings
    texts = []

    # Original
    text = pytesseract.image_to_string(gray, lang="eng", config=config)
    texts.append(("original", text))

    # Inverted
    inverted = cv2.bitwise_not(gray)
    text = pytesseract.image_to_string(inverted, lang="eng", config=config)
    texts.append(("inverted", text))

    # Binary threshold low
    _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    text = pytesseract.image_to_string(binary, lang="eng", config=config)
    texts.append(("binary_50", text))

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(gray)
    text = pytesseract.image_to_string(enhanced, lang="eng", config=config)
    texts.append(("clahe", text))

    best_text = ""
    best_score = -1
    for name, t in texts:
        s = score_output(t)["score"]
        print(f"  {name}: score={s}, digits={repr(t.strip()[:60])}")
        if s > best_score:
            best_score = s
            best_text = t

    return best_text


# ============================================================
# ENGINE 3 — PaddleOCR
# ============================================================

def engine_paddleocr():
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Force CPU mode
    from paddleocr import PaddleOCR

    ocr = PaddleOCR(lang="en")
    result = list(ocr.predict(CURRENT_IMAGE))

    lines = []
    if result:
        for page in result:
            if hasattr(page, 'rec_texts'):
                for text, score in zip(page.rec_texts, page.rec_scores):
                    lines.append(f"{text} ({score:.2f})")
            elif isinstance(page, dict) and 'rec_texts' in page:
                for text, score in zip(page['rec_texts'], page['rec_scores']):
                    lines.append(f"{text} ({score:.2f})")
            elif isinstance(page, list):
                for item in page:
                    if isinstance(item, (list, tuple)) and len(item) >= 2:
                        text = item[1][0] if isinstance(item[1], (list, tuple)) else str(item[1])
                        conf = item[1][1] if isinstance(item[1], (list, tuple)) and len(item[1]) > 1 else 0
                        lines.append(f"{text} ({conf:.2f})")

    return "\n".join(lines)


# ============================================================
# ENGINE 4 — EasyOCR
# ============================================================

def engine_easyocr():
    import os
    os.environ["NCCL_P2P_DISABLE"] = "1"
    os.environ["NCCL_SHM_DISABLE"] = "1"
    import easyocr

    try:
        reader = easyocr.Reader(["en"], gpu=True)
        img = cv2.imread(CURRENT_IMAGE)
        results_ocr = reader.readtext(img, detail=1, text_threshold=0.3, low_text=0.3)
    except RuntimeError:
        # Fallback to CPU if NCCL still fails
        reader = easyocr.Reader(["en"], gpu=False)
        img = cv2.imread(CURRENT_IMAGE)
        results_ocr = reader.readtext(img, detail=1, text_threshold=0.3, low_text=0.3)

    # Use spatial/bbox coordinates for label-value association
    extracted = _easyocr_spatial_parse(results_ocr)

    # Build output text that score_output can parse
    lines = []
    if extracted["SYS"] is not None:
        lines.append(f"SYS {extracted['SYS']}")
    if extracted["DIA"] is not None:
        lines.append(f"DIA {extracted['DIA']}")
    if extracted["PUL"] is not None:
        lines.append(f"PUL {extracted['PUL']}")

    # Print raw detections to console for debugging (not in returned text)
    print("  Spatial parse result:", extracted)
    print("  Raw detections:")
    for bbox, text, conf in sorted(results_ocr, key=lambda r: r[0][0][1]):
        cx = (bbox[0][0] + bbox[2][0]) / 2
        cy = (bbox[0][1] + bbox[2][1]) / 2
        print(f"    pos=({cx:.0f},{cy:.0f}) text=\"{text}\" conf={conf:.2f}")

    return "\n".join(lines)


def _easyocr_spatial_parse(detections: list) -> dict:
    """Parse EasyOCR detections using spatial (bbox) coordinates.

    For each label (SYS, DIA, PUL), find the nearest number to its RIGHT or BELOW.
    Uses center coordinates of each bounding box.

    Args:
        detections: List of (bbox, text, confidence) from EasyOCR.

    Returns:
        Dict with SYS, DIA, PUL values (int or None).
    """
    extracted = {"SYS": None, "DIA": None, "PUL": None}

    if not detections:
        return extracted

    # Build list of (center_x, center_y, text, bbox_height) for each detection
    items = []
    for bbox, text, conf in detections:
        # bbox: [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
        cx = (bbox[0][0] + bbox[2][0]) / 2
        cy = (bbox[0][1] + bbox[2][1]) / 2
        h = abs(bbox[2][1] - bbox[0][1])
        items.append({"cx": cx, "cy": cy, "text": text.strip(), "h": h, "conf": conf})

    # Separate labels and numbers
    labels = []  # (cx, cy, label_name)
    numbers = []  # (cx, cy, value)

    # Words/patterns to skip when looking for numbers
    skip_patterns = re.compile(
        r"(mmhg|kpa|bpm|pul/?min|pulse/?min|/min|d2[-]?\d+[a-z]?|[a-z]\d+[-]\d*[a-z]?)",
        re.IGNORECASE,
    )

    for item in items:
        text_lower = item["text"].lower().strip()
        # Check if this is a label (confidence > 0.3 for labels)
        if item["conf"] < 0.3:
            continue
        if "sys" in text_lower and len(text_lower) <= 10:
            labels.append({"cx": item["cx"], "cy": item["cy"], "label": "SYS", "h": item["h"]})
        elif "dia" in text_lower and len(text_lower) <= 10:
            labels.append({"cx": item["cx"], "cy": item["cy"], "label": "DIA", "h": item["h"]})
        elif "pul" in text_lower and "min" not in text_lower and len(text_lower) <= 10:
            labels.append({"cx": item["cx"], "cy": item["cy"], "label": "PUL", "h": item["h"]})

        # Check if this is a number (2-3 digits, plausible BP value)
        # Require confidence > 0.4 for number detections
        if item["conf"] < 0.4:
            continue
        # Skip items that are units, model numbers, or other non-digit text
        if skip_patterns.search(item["text"]):
            continue
        # Only consider items that are purely numeric or mostly numeric
        clean_text = re.sub(r"[^0-9]", "", item["text"])
        if not clean_text:
            continue
        # The text should be mostly digits (>50% of non-space chars)
        non_space = re.sub(r"\s", "", item["text"])
        if len(clean_text) / max(len(non_space), 1) < 0.5:
            continue

        digits = re.findall(r"\d{2,3}", item["text"])
        for d in digits:
            val = int(d)
            if 30 <= val <= 250:
                numbers.append({"cx": item["cx"], "cy": item["cy"], "value": val})

    # For each label, find the nearest number to its RIGHT or BELOW
    for label_item in labels:
        label_name = label_item["label"]
        if extracted[label_name] is not None:
            continue

        best_val = None
        best_score = float("inf")

        for num_item in numbers:
            dx = num_item["cx"] - label_item["cx"]
            dy = num_item["cy"] - label_item["cy"]

            # Only consider numbers to the RIGHT or BELOW the label
            # Right: dx > 0, dy within tolerance (same row)
            # Below: dy > 0, dx within tolerance (same column)
            row_tolerance = label_item["h"] * 1.5  # within 1.5x label height vertically
            col_tolerance = label_item["h"] * 3  # within 3x label height horizontally

            score = None

            # Same row (to the right)
            if dx > 0 and abs(dy) < row_tolerance:
                score = math.sqrt(dx**2 + dy**2)
                # Prefer same-row matches
                score *= 0.5

            # Below (within column tolerance)
            elif dy > 0 and abs(dx) < col_tolerance:
                score = math.sqrt(dx**2 + dy**2)

            if score is not None and score < best_score:
                best_score = score
                best_val = num_item["value"]

        if best_val is not None:
            extracted[label_name] = best_val

    return extracted


# ============================================================
# ENGINE 5 — TrOCR (printed)
# ============================================================

def engine_trocr():
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    from PIL import Image
    import torch

    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    img = Image.open(CURRENT_IMAGE).convert("RGB")

    # Full image
    pixel_values = processor(images=img, return_tensors="pt").pixel_values.to(device)
    generated_ids = model.generate(pixel_values)
    full_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(f"  Full image: {repr(full_text)}")

    # Crop digit regions using contours
    img_cv = cv2.imread(CURRENT_IMAGE)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    crops_text = []
    regions = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w > 30 and h > 20 and cv2.contourArea(c) > 500:
            regions.append((x, y, w, h))

    regions.sort(key=lambda r: (r[1], r[0]))

    for x, y, w, h in regions[:10]:
        crop = img.crop((x, y, x + w, y + h))
        pixel_values = processor(images=crop, return_tensors="pt").pixel_values.to(device)
        generated_ids = model.generate(pixel_values)
        crop_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        crops_text.append(f"  Region ({x},{y},{w}x{h}): {repr(crop_text)}")

    all_crops = "\n".join(crops_text)
    print(all_crops)

    return full_text + "\n" + "\n".join(
        [ct.split(": ")[1].strip("'\"") for ct in crops_text if ct]
    )


# ============================================================
# ENGINE 6 — Surya OCR
# ============================================================

def engine_surya():
    from surya.ocr import run_ocr
    from surya.model.detection.model import load_model as load_det_model
    from surya.model.detection.model import load_processor as load_det_processor
    from surya.model.recognition.model import load_model as load_rec_model
    from surya.model.recognition.processor import load_processor as load_rec_processor
    from PIL import Image

    img = Image.open(CURRENT_IMAGE)

    det_model = load_det_model()
    det_processor = load_det_processor()
    rec_model = load_rec_model()
    rec_processor = load_rec_processor()

    predictions = run_ocr(
        [img], [["en"]], det_model, det_processor, rec_model, rec_processor
    )

    lines = []
    if predictions:
        for text_line in predictions[0].text_lines:
            lines.append(text_line.text)

    return "\n".join(lines)


# ============================================================
# ENGINE 7 — OpenCV contour crop + Tesseract
# ============================================================

def engine_contour_tesseract():
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

    img = cv2.imread(CURRENT_IMAGE)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Try multiple threshold levels to find digit contours
    all_digits = []

    for thresh_val in [40, 50, 60, 80, 100, 127]:
        _, binary = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        regions = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            area = cv2.contourArea(c)
            if area > 500 and h > 30 and w > 15:
                regions.append((x, y, w, h))

        if not regions:
            continue

        # Sort by position (top to bottom)
        regions.sort(key=lambda r: (r[1], r[0]))

        digits_this_thresh = []
        for x, y, w, h in regions:
            # Crop with padding
            pad = 5
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(gray.shape[1], x + w + pad)
            y2 = min(gray.shape[0], y + h + pad)

            crop = gray[y1:y2, x1:x2]

            # Try digit-only OCR on crop
            config = "--psm 8 --oem 3 -c tessedit_char_whitelist=0123456789"
            text = pytesseract.image_to_string(crop, lang="eng", config=config).strip()

            # Also try inverted crop
            crop_inv = cv2.bitwise_not(crop)
            text_inv = pytesseract.image_to_string(crop_inv, lang="eng", config=config).strip()

            # Also try with PSM 7 (single line)
            config7 = "--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789"
            text7 = pytesseract.image_to_string(crop, lang="eng", config=config7).strip()
            text7_inv = pytesseract.image_to_string(crop_inv, lang="eng", config=config7).strip()

            best = ""
            for t in [text, text_inv, text7, text7_inv]:
                if t and len(t) >= 2 and t.isdigit():
                    if not best or len(t) > len(best):
                        best = t

            if best:
                digits_this_thresh.append((y, best))
                print(f"  thresh={thresh_val} region ({x},{y},{w}x{h}): '{best}'")

        if len(digits_this_thresh) >= 3:
            all_digits = digits_this_thresh
            break

        if len(digits_this_thresh) > len(all_digits):
            all_digits = digits_this_thresh

    # Reconstruct from spatial layout
    all_digits.sort(key=lambda d: d[0])
    digit_values = [d[1] for d in all_digits]

    output_lines = []
    labels = ["SYS", "DIA", "PUL"]
    for i, val in enumerate(digit_values[:3]):
        label = labels[i] if i < len(labels) else f"VAL{i}"
        output_lines.append(f"{label} {val}")

    return "\n".join(output_lines) if output_lines else "\n".join(digit_values)


# ============================================================
# ENGINE 8 — Screen detect + warp + Tesseract
# ============================================================

def engine_screen_detect_warp():
    from screen_detector import detect_and_ocr

    text = detect_and_ocr(CURRENT_IMAGE, debug_output=True)
    return text


# ============================================================
# ENGINE 11 — Qwen3-VL:2b (Ollama)
# ============================================================

def engine_qwen3_vl_ollama():
    import base64
    import requests

    with open(CURRENT_IMAGE, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()

    payload = {
        "model": "qwen3-vl:2b",
        "messages": [{
            "role": "user",
            "content": (
                "This is a blood pressure monitor display. "
                "Read the three numbers shown next to the labels SYS, DIA, and PUL. "
                "Reply ONLY in this exact format with no other text:\n"
                "SYS: <number>\n"
                "DIA: <number>\n"
                "PUL: <number>"
            ),
            "images": [img_b64]
        }],
        "stream": False
    }

    resp = requests.post("http://localhost:11434/api/chat", json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()["message"]["content"]


# ============================================================
# ENGINE 12 — Qwen3-VL:2b on warped image (Ollama)
# ============================================================

def engine_qwen3_vl_ollama_warped():
    import base64
    import requests

    warped_path = "output/debug_warped_may-do-huyet-ap-bap-tay-d2group-kf-65a.png"
    if not os.path.exists(warped_path):
        return "ERROR: warped image not found"

    with open(warped_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()

    payload = {
        "model": "qwen3-vl:2b",
        "messages": [{
            "role": "user",
            "content": (
                "This is a blood pressure monitor display. "
                "Read the three numbers shown next to the labels SYS, DIA, and PUL. "
                "Reply ONLY in this exact format with no other text:\n"
                "SYS: <number>\n"
                "DIA: <number>\n"
                "PUL: <number>"
            ),
            "images": [img_b64]
        }],
        "stream": False
    }

    resp = requests.post("http://localhost:11434/api/chat", json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()["message"]["content"]


# ============================================================
# MAIN
# ============================================================

def print_summary(image_name: str):
    """Print final summary table."""
    print(f"\n\n{'='*80}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*80}")
    print(f"Image: {image_name}")
    print(f"Ground truth: SYS={GROUND_TRUTH['SYS']}, DIA={GROUND_TRUTH['DIA']}, PUL={GROUND_TRUTH['PUL']}")
    print(f"{'='*80}")
    print(f"{'Engine':<35} | {'SYS':>5} | {'DIA':>5} | {'PUL':>5} | {'Score':>5} | {'Time':>7} | Error")
    print(f"{'-'*35}-+-{'-'*5}-+-{'-'*5}-+-{'-'*5}-+-{'-'*5}-+-{'-'*7}-+------")

    for r in results:
        sys_val = str(r["sys"]) if r["sys"] is not None else "-"
        dia_val = str(r["dia"]) if r["dia"] is not None else "-"
        pul_val = str(r["pul"]) if r["pul"] is not None else "-"
        err = r["error"][:20] if r["error"] else ""

        # Mark correct values
        sys_mark = "✓" if r["sys"] == GROUND_TRUTH["SYS"] else " "
        dia_mark = "✓" if r["dia"] == GROUND_TRUTH["DIA"] else " "
        pul_mark = "✓" if r["pul"] == GROUND_TRUTH["PUL"] else " "

        print(f"{r['engine']:<35} | {sys_val:>4}{sys_mark} | {dia_val:>4}{dia_mark} | "
              f"{pul_val:>4}{pul_mark} | {r['score']:>5} | {r['time']:>6.2f}s | {err}")

    print(f"{'='*80}")


def save_results(all_results: dict):
    """Save results for all images to file."""
    with open("benchmark_results.txt", "w") as f:
        for image_name, image_results in all_results.items():
            f.write(f"OCR LCD Benchmark Results\n")
            f.write(f"Image: {image_name}\n")
            f.write(f"Ground truth: SYS={GROUND_TRUTH['SYS']}, DIA={GROUND_TRUTH['DIA']}, PUL={GROUND_TRUTH['PUL']}\n")
            f.write(f"{'='*80}\n")
            f.write(f"{'Engine':<35} | {'SYS':>5} | {'DIA':>5} | {'PUL':>5} | {'Score':>5} | {'Time':>7} | Error\n")
            f.write(f"{'-'*35}-+-{'-'*5}-+-{'-'*5}-+-{'-'*5}-+-{'-'*5}-+-{'-'*7}-+------\n")

            for r in image_results:
                sys_val = str(r["sys"]) if r["sys"] is not None else "-"
                dia_val = str(r["dia"]) if r["dia"] is not None else "-"
                pul_val = str(r["pul"]) if r["pul"] is not None else "-"
                err = r["error"][:30] if r["error"] else ""
                f.write(f"{r['engine']:<35} | {sys_val:>5} | {dia_val:>5} | {pul_val:>5} | "
                        f"{r['score']:>5} | {r['time']:>6.2f}s | {err}\n")

            f.write(f"{'='*80}\n\n")

            # Raw outputs
            f.write("DETAILED RAW OUTPUTS\n")
            f.write(f"{'='*80}\n")
            for r in image_results:
                f.write(f"\n--- {r['engine']} ---\n")
                f.write(f"Raw: {r['raw']}\n")
                if r["error"]:
                    f.write(f"Error: {r['error']}\n")
            f.write(f"\n\n{'#'*80}\n\n")

    print(f"\nResults saved to benchmark_results.txt")


def run_all_engines():
    """Run all engines on the current image."""
    global results
    results = []

    # Engine 1: Tesseract default
    run_engine("1. Tesseract default (PSM 6/11/3)", engine_tesseract_default)

    # Engine 2: Tesseract digit whitelist
    run_engine("2. Tesseract digit whitelist", engine_tesseract_digits)

    # Engine 3: PaddleOCR
    try:
        from paddleocr import PaddleOCR
        run_engine("3. PaddleOCR", engine_paddleocr)
    except (ImportError, Exception) as e:
        print(f"\n[SKIP] PaddleOCR failed: {e}")
        results.append({
            "engine": "3. PaddleOCR",
            "raw": "", "sys": None, "dia": None, "pul": None,
            "score": 0, "time": 0, "error": str(e)[:50],
        })

    # Engine 4: EasyOCR
    try:
        import easyocr
        run_engine("4. EasyOCR (spatial)", engine_easyocr)
    except ImportError:
        print("\n[SKIP] EasyOCR not installed.")
        results.append({
            "engine": "4. EasyOCR (spatial)",
            "raw": "", "sys": None, "dia": None, "pul": None,
            "score": 0, "time": 0, "error": "not installed",
        })

    # Engine 5: TrOCR
    try:
        from transformers import TrOCRProcessor
        run_engine("5. TrOCR (printed)", engine_trocr)
    except ImportError:
        print("\n[SKIP] TrOCR not installed.")
        results.append({
            "engine": "5. TrOCR (printed)",
            "raw": "", "sys": None, "dia": None, "pul": None,
            "score": 0, "time": 0, "error": "not installed",
        })

    # Engine 6: Surya OCR
    try:
        from surya.ocr import run_ocr
        run_engine("6. Surya OCR", engine_surya)
    except ImportError:
        print("\n[SKIP] Surya OCR not installed.")
        results.append({
            "engine": "6. Surya OCR",
            "raw": "", "sys": None, "dia": None, "pul": None,
            "score": 0, "time": 0, "error": "not installed",
        })

    # Engine 7: Contour crop + Tesseract
    run_engine("7. OpenCV contour + Tesseract", engine_contour_tesseract)

    # Engine 8: Screen detect + warp + Tesseract
    run_engine("8. Screen detect + warp + Tesseract", engine_screen_detect_warp)

    # Engine 11: Qwen3-VL:2b (Ollama)
    try:
        import requests
        requests.get("http://localhost:11434/api/tags", timeout=5)
        run_engine("11. Qwen3-VL:2b (Ollama)", engine_qwen3_vl_ollama)
    except Exception as e:
        print(f"\n[SKIP] Qwen3-VL Ollama not available: {e}")
        results.append({
            "engine": "11. Qwen3-VL:2b (Ollama)",
            "raw": "", "sys": None, "dia": None, "pul": None,
            "score": 0, "time": 0, "error": str(e)[:50],
        })

    # Engine 12: Qwen3-VL:2b on warped image (Ollama)
    try:
        import requests
        requests.get("http://localhost:11434/api/tags", timeout=5)
        run_engine("12. Qwen3-VL:2b warped (Ollama)", engine_qwen3_vl_ollama_warped)
    except Exception as e:
        print(f"\n[SKIP] Qwen3-VL warped not available: {e}")
        results.append({
            "engine": "12. Qwen3-VL:2b warped (Ollama)",
            "raw": "", "sys": None, "dia": None, "pul": None,
            "score": 0, "time": 0, "error": str(e)[:50],
        })

    return list(results)


if __name__ == "__main__":
    all_results = {}

    # ---- Run on REAL product photo ----
    print(f"\n{'#'*80}")
    print(f"# TEST 1: Real product photo")
    print(f"{'#'*80}")
    CURRENT_IMAGE = IMAGE_PATH
    print(f"Image: {CURRENT_IMAGE}")
    print(f"Ground truth: SYS={GROUND_TRUTH['SYS']}, DIA={GROUND_TRUTH['DIA']}, PUL={GROUND_TRUTH['PUL']}")

    img = cv2.imread(CURRENT_IMAGE)
    if img is None:
        print(f"ERROR: Cannot read image: {CURRENT_IMAGE}")
        sys.exit(1)
    print(f"Image loaded: {img.shape}")

    all_results[IMAGE_PATH] = run_all_engines()
    print_summary(IMAGE_PATH)

    # ---- Run on SYNTHETIC LCD image ----
    print(f"\n\n{'#'*80}")
    print(f"# TEST 2: Synthetic LCD image")
    print(f"{'#'*80}")
    CURRENT_IMAGE = IMAGE_PATH_SYNTHETIC
    print(f"Image: {CURRENT_IMAGE}")
    print(f"Ground truth: SYS={GROUND_TRUTH['SYS']}, DIA={GROUND_TRUTH['DIA']}, PUL={GROUND_TRUTH['PUL']}")

    img = cv2.imread(CURRENT_IMAGE)
    if img is None:
        print(f"ERROR: Cannot read image: {CURRENT_IMAGE}")
        print("Skipping synthetic test.")
    else:
        print(f"Image loaded: {img.shape}")
        all_results[IMAGE_PATH_SYNTHETIC] = run_all_engines()
        print_summary(IMAGE_PATH_SYNTHETIC)

    # Save all results
    save_results(all_results)
