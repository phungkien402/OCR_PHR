"""Regex-based parser for extracting vital signs from OCR text."""

import logging
import re
import unicodedata

from .config import FIELD_KEYWORDS

logger = logging.getLogger(__name__)

# English labels commonly found on LCD blood pressure monitors
LCD_LABELS = {
    "sys": "huyet_ap.tam_thu",
    "systolic": "huyet_ap.tam_thu",
    "dia": "huyet_ap.tam_truong",
    "diastolic": "huyet_ap.tam_truong",
    "pul": "mach",
    "pulse": "mach",
    "pul/min": "mach",
    "pulse/min": "mach",
}

# Unit labels to capture as metadata (not affecting values)
UNIT_LABELS = {"mmhg", "kpa", "bpm", "/min", "min"}


def parse_vitals(raw_text: str) -> dict:
    """Parse vital signs from raw OCR text.

    Args:
        raw_text: Raw text extracted by OCR engine.

    Returns:
        Dictionary with extracted vital sign values.
        Missing fields are set to None.
    """
    logger.info("Parsing vital signs from OCR text")
    logger.debug("Raw text:\n%s", raw_text)

    # Normalize text for matching
    normalized = _normalize_text(raw_text)

    # Try LCD label parsing first (for BP monitor images)
    lcd_vitals = _parse_lcd_labels(normalized)

    # Then try Vietnamese keyword parsing
    keyword_vitals = {
        "mach": _extract_integer(normalized, "mach"),
        "nhiet_do": _extract_float(normalized, "nhiet_do"),
        "huyet_ap": _extract_blood_pressure(normalized),
        "nhip_tho": _extract_integer(normalized, "nhip_tho"),
        "can_nang": _extract_float(normalized, "can_nang"),
        "chieu_cao": _extract_integer(normalized, "chieu_cao"),
        "spo2": _extract_integer(normalized, "spo2"),
    }

    # Merge: LCD results take priority where available
    vitals = _merge_vitals(keyword_vitals, lcd_vitals)

    logger.info("Parsed vitals: %s", vitals)
    return vitals


def _parse_lcd_labels(text: str) -> dict:
    """Parse LCD blood pressure monitor labels (SYS, DIA, PUL).

    Uses two strategies:
    1. Same-line pattern: "SYS 128", "DIA 78", "PUL 72"
    2. Line-by-line sequential: label on one line, value on next line

    Args:
        text: Normalized OCR text.

    Returns:
        Partial vitals dict with values found via LCD labels.
    """
    result = {
        "mach": None,
        "huyet_ap": None,
    }

    tam_thu = None
    tam_truong = None
    units = []

    # Strategy 1: Same-line pattern matching
    # "SYS 128", "SYS:128", "SYS.128"
    for label, field in LCD_LABELS.items():
        pattern = re.escape(label) + r"[\s.:;=]*(\d{2,3})"
        match = re.search(pattern, text)
        if match:
            value = int(match.group(1))
            if field == "huyet_ap.tam_thu":
                tam_thu = value
                logger.debug("LCD same-line: SYS = %d", value)
            elif field == "huyet_ap.tam_truong":
                tam_truong = value
                logger.debug("LCD same-line: DIA = %d", value)
            elif field == "mach":
                result["mach"] = value
                logger.debug("LCD same-line: PUL = %d", value)

    # Strategy 2: Line-by-line sequential matching
    # Label on one line, digit value on the next non-empty line
    if tam_thu is None or tam_truong is None or result["mach"] is None:
        line_results = _parse_lcd_lines(text)
        if tam_thu is None and line_results.get("tam_thu") is not None:
            tam_thu = line_results["tam_thu"]
            logger.debug("LCD line-seq: SYS = %d", tam_thu)
        if tam_truong is None and line_results.get("tam_truong") is not None:
            tam_truong = line_results["tam_truong"]
            logger.debug("LCD line-seq: DIA = %d", tam_truong)
        if result["mach"] is None and line_results.get("mach") is not None:
            result["mach"] = line_results["mach"]
            logger.debug("LCD line-seq: PUL = %d", result["mach"])

    # Capture unit metadata
    for unit in UNIT_LABELS:
        if unit in text:
            units.append(unit)

    if tam_thu is not None or tam_truong is not None:
        result["huyet_ap"] = {"tam_thu": tam_thu, "tam_truong": tam_truong}

    if units:
        result["_units"] = units
        logger.debug("LCD units detected: %s", units)

    return result


def _parse_lcd_lines(text: str) -> dict:
    """Parse LCD values using line-by-line sequential matching.

    When OCR returns labels and values on separate lines, associate them
    by proximity: the number on the next line after a label belongs to it.

    Args:
        text: Normalized OCR text.

    Returns:
        Dict with tam_thu, tam_truong, mach values (or None).
    """
    lines = text.split("\n")
    result = {"tam_thu": None, "tam_truong": None, "mach": None}

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Check if this line contains a label
        label_field = _identify_lcd_label(line)

        if label_field is not None:
            # Look for digit value: first check same line after label
            # Then check subsequent lines
            value = _extract_digit_from_line(line, label_field)

            if value is None:
                # Search next non-empty lines for a digit
                for j in range(i + 1, min(i + 3, len(lines))):
                    next_line = lines[j].strip()
                    if not next_line:
                        continue
                    # Skip lines that are model numbers (e.g. D2-65A)
                    if re.match(r"^[A-Za-z]\d+[-]?\w*$", next_line):
                        continue
                    # Skip lines that are other labels
                    if _identify_lcd_label(next_line):
                        break
                    # Check if next line is purely a number
                    digit_match = re.search(r"^(\d{2,3})$", next_line)
                    if digit_match:
                        value = int(digit_match.group(1))
                        break
                    # Check if next line contains a number (possibly with units)
                    digit_match = re.search(r"(\d{2,3})", next_line)
                    if digit_match and not _identify_lcd_label(next_line):
                        # Make sure it's not part of a model number
                        if not re.search(r"[A-Za-z]\d+[-]?\w*", next_line):
                            value = int(digit_match.group(1))
                            break

            if value is not None:
                if label_field == "tam_thu":
                    result["tam_thu"] = value
                elif label_field == "tam_truong":
                    result["tam_truong"] = value
                elif label_field == "mach":
                    result["mach"] = value

        i += 1

    return result


def _identify_lcd_label(line: str) -> object:
    """Identify if a line contains an LCD label.

    Args:
        line: A single line of text (already lowercase).

    Returns:
        Field name ("tam_thu", "tam_truong", "mach") or None.
    """
    line_clean = line.strip().lower()

    # Check for SYS variants
    if re.search(r"\bsys\b|systolic", line_clean):
        return "tam_thu"

    # Check for DIA variants
    if re.search(r"\bdia\b|diastolic", line_clean):
        return "tam_truong"

    # Check for PUL variants
    if re.search(r"\bpul\b|\bpulse\b|pul/min|pulse/min", line_clean):
        return "mach"

    return None


def _extract_digit_from_line(line: str, label_field: str) -> object:
    """Try to extract a digit value from the same line as a label.

    Args:
        line: The line containing the label.
        label_field: Which field the label represents.

    Returns:
        Integer value or None.
    """
    # Remove the label text and look for remaining digits
    cleaned = re.sub(r"(sys|dia|pul|pulse|systolic|diastolic|/min)", "", line, flags=re.IGNORECASE)
    # Exclude model numbers like D2-65A
    cleaned = re.sub(r"[a-zA-Z]\d+-?\d*[a-zA-Z]?", "", cleaned)
    match = re.search(r"(\d{2,3})", cleaned)
    if match:
        return int(match.group(1))
    return None


def _merge_vitals(keyword_vitals: dict, lcd_vitals: dict) -> dict:
    """Merge keyword-parsed vitals with LCD-parsed vitals.

    LCD results take priority where they provide non-None values.

    Args:
        keyword_vitals: Results from Vietnamese keyword parsing.
        lcd_vitals: Results from LCD label parsing.

    Returns:
        Merged vitals dictionary.
    """
    vitals = keyword_vitals.copy()

    # LCD mach overrides keyword mach
    if lcd_vitals.get("mach") is not None:
        vitals["mach"] = lcd_vitals["mach"]

    # LCD blood pressure overrides keyword blood pressure
    lcd_bp = lcd_vitals.get("huyet_ap")
    if lcd_bp is not None:
        if vitals["huyet_ap"] is None:
            vitals["huyet_ap"] = {}
        if lcd_bp.get("tam_thu") is not None:
            vitals["huyet_ap"]["tam_thu"] = lcd_bp["tam_thu"]
        if lcd_bp.get("tam_truong") is not None:
            vitals["huyet_ap"]["tam_truong"] = lcd_bp["tam_truong"]
        # Clean up None sub-fields
        if vitals["huyet_ap"].get("tam_thu") is None and vitals["huyet_ap"].get("tam_truong") is None:
            vitals["huyet_ap"] = None

    # Store unit metadata if present
    if "_units" in lcd_vitals:
        vitals["_units"] = lcd_vitals["_units"]

    return vitals


def _normalize_text(text: str) -> str:
    """Normalize text for keyword matching (lowercase, preserve diacritics)."""
    return text.lower().strip()


def _remove_diacritics(text: str) -> str:
    """Remove Vietnamese diacritics for fuzzy matching."""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def _find_keyword_position(text: str, field: str) -> int:
    """Find the position of a field keyword in the text.

    Tries exact match first, then diacritic-stripped match,
    then fuzzy match (ignoring spaces within keywords).

    Returns:
        Position (end of keyword) or -1 if not found.
    """
    keywords = FIELD_KEYWORDS.get(field, [])

    for keyword in keywords:
        # Exact match (case-insensitive)
        pos = text.find(keyword.lower())
        if pos != -1:
            return pos + len(keyword)

    # Try without diacritics
    text_no_diac = _remove_diacritics(text)
    for keyword in keywords:
        keyword_no_diac = _remove_diacritics(keyword.lower())
        pos = text_no_diac.find(keyword_no_diac)
        if pos != -1:
            return pos + len(keyword_no_diac)

    # Try without diacritics AND without spaces (OCR often merges words)
    text_no_diac_no_space = text_no_diac.replace(" ", "")
    for keyword in keywords:
        keyword_no_diac = _remove_diacritics(keyword.lower()).replace(" ", "")
        pos = text_no_diac_no_space.find(keyword_no_diac)
        if pos != -1:
            # Map back to approximate position in original text
            return _map_position_back(text_no_diac, pos + len(keyword_no_diac))

    return -1


def _map_position_back(text: str, compact_pos: int) -> int:
    """Map a position from space-stripped text back to original text.

    Args:
        text: Original text (with spaces).
        compact_pos: Position in space-stripped version.

    Returns:
        Approximate position in original text.
    """
    count = 0
    for i, ch in enumerate(text):
        if ch != " ":
            count += 1
        if count >= compact_pos:
            return i + 1
    return len(text)


def _extract_nearest_number(text: str, start_pos: int, is_float: bool = False) -> object:
    """Extract the nearest number after a given position in text.

    Args:
        text: The text to search in.
        start_pos: Position to start searching from.
        is_float: If True, look for decimal numbers.

    Returns:
        Extracted number (int or float) or None.
    """
    # Search in a window after the keyword
    search_window = text[start_pos:start_pos + 50]

    # Strip leading punctuation/separators that OCR might produce (.:, etc.)
    search_window = re.sub(r"^[\s.:,;=]+", "", search_window)

    if is_float:
        # Match decimal numbers like 37.5, 37,5, or plain integers
        match = re.search(r"(\d+[.,]\d+|\d+)", search_window)
        if match:
            value_str = match.group(1).replace(",", ".")
            try:
                return float(value_str)
            except ValueError:
                return None
    else:
        # Match integers
        match = re.search(r"(\d+)", search_window)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                return None

    return None


def _extract_integer(text: str, field: str) -> object:
    """Extract an integer value for a given field."""
    pos = _find_keyword_position(text, field)
    if pos == -1:
        logger.debug("Keyword not found for field: %s", field)
        return None

    value = _extract_nearest_number(text, pos, is_float=False)
    if value is not None:
        logger.debug("Extracted %s = %d", field, value)
    return value


def _extract_float(text: str, field: str) -> object:
    """Extract a float value for a given field."""
    pos = _find_keyword_position(text, field)
    if pos == -1:
        logger.debug("Keyword not found for field: %s", field)
        return None

    value = _extract_nearest_number(text, pos, is_float=True)
    if value is not None:
        logger.debug("Extracted %s = %.1f", field, value)
    return value


def _extract_blood_pressure(text: str) -> object:
    """Extract blood pressure (systolic/diastolic) from text.

    Looks for pattern NNN/NN or NNN/NNN near blood pressure keywords.
    Falls back to searching the entire text for BP-like patterns.

    Returns:
        Dict with tam_thu and tam_truong, or None if not found.
    """
    pos = _find_keyword_position(text, "huyet_ap")

    # Build search candidates: near keyword first, then full text
    search_areas = []
    if pos != -1:
        search_areas.append(text[pos:pos + 50])
    search_areas.append(text)

    for search_window in search_areas:
        # Match blood pressure pattern: 2-3 digits / 2-3 digits
        match = re.search(r"(\d{2,3})\s*/\s*(\d{2,3})", search_window)
        if match:
            tam_thu = int(match.group(1))
            tam_truong = int(match.group(2))
            logger.debug("Extracted huyet_ap = %d/%d", tam_thu, tam_truong)
            return {"tam_thu": tam_thu, "tam_truong": tam_truong}

        # Try alternative separators (dash, backslash)
        match = re.search(r"(\d{2,3})\s*[-\\]\s*(\d{2,3})", search_window)
        if match:
            tam_thu = int(match.group(1))
            tam_truong = int(match.group(2))
            logger.debug("Extracted huyet_ap = %d/%d (alt separator)", tam_thu, tam_truong)
            return {"tam_thu": tam_thu, "tam_truong": tam_truong}

    logger.debug("Blood pressure pattern not found")
    return None
