"""LCD screen detection and perspective correction.

Detects the rectangular LCD screen region in a blood pressure monitor image,
applies perspective transform to get a flat front-on view, then runs OCR
on the corrected image.
"""

import logging
import os

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def detect_screen(image: np.ndarray) -> np.ndarray | None:
    """Detect the LCD screen region as a 4-corner polygon.

    Args:
        image: BGR image as numpy array.

    Returns:
        4x2 numpy array of corner points (ordered: TL, TR, BR, BL),
        or None if no screen detected.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_area = gray.shape[0] * gray.shape[1]
    min_screen_area = img_area * 0.10  # Screen must be at least 10% of image

    # Apply blur to reduce noise before edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Try multiple Canny thresholds to find the screen
    for low_thresh, high_thresh in [(30, 100), (50, 150), (20, 80), (70, 200)]:
        edges = cv2.Canny(blurred, low_thresh, high_thresh)

        # Dilate edges to close gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Sort by area (largest first)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for contour in contours[:10]:
            area = cv2.contourArea(contour)
            if area < min_screen_area:
                continue

            # Approximate polygon
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

            # Look for quadrilateral (4 corners)
            if len(approx) == 4:
                corners = approx.reshape(4, 2)
                ordered = _order_corners(corners)
                logger.info(
                    "Screen detected (thresh=%d/%d): area=%.0f, corners=%s",
                    low_thresh, high_thresh, area, ordered.tolist()
                )
                return ordered

    # Fallback: try adaptive threshold + contour approach
    screen = _detect_screen_adaptive(gray, min_screen_area)
    if screen is not None:
        return screen

    logger.warning("No LCD screen region detected")
    return None


def _detect_screen_adaptive(gray: np.ndarray, min_area: float) -> np.ndarray | None:
    """Fallback screen detection using adaptive thresholding.

    Args:
        gray: Grayscale image.
        min_area: Minimum area threshold.

    Returns:
        4x2 corner array or None.
    """
    # Try Otsu threshold
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphological close to fill the screen region
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for contour in contours[:5]:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        peri = cv2.arcLength(contour, True)
        # More lenient approximation
        for epsilon in [0.02, 0.03, 0.05]:
            approx = cv2.approxPolyDP(contour, epsilon * peri, True)
            if len(approx) == 4:
                corners = approx.reshape(4, 2)
                ordered = _order_corners(corners)
                logger.info("Screen detected (adaptive, eps=%.2f): area=%.0f", epsilon, area)
                return ordered

        # If we can't get exactly 4 corners, use bounding rect
        if area > min_area * 1.5:
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            corners = np.int32(box)
            ordered = _order_corners(corners)
            logger.info("Screen detected (minAreaRect): area=%.0f", area)
            return ordered

    return None


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """Order 4 corners as: top-left, top-right, bottom-right, bottom-left.

    Args:
        pts: 4x2 array of corner points.

    Returns:
        4x2 array with ordered corners.
    """
    pts = pts.astype(np.float32)

    # Sort by sum (x+y): smallest = top-left, largest = bottom-right
    s = pts.sum(axis=1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]

    # Sort by difference (y-x): smallest = top-right, largest = bottom-left
    d = np.diff(pts, axis=1).flatten()
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]

    return np.array([tl, tr, br, bl], dtype=np.float32)


def warp_screen(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Apply perspective transform to get a flat view of the screen.

    Args:
        image: Original BGR image.
        corners: 4x2 array of screen corners (TL, TR, BR, BL).

    Returns:
        Warped image (flat, rectangular view of the screen).
    """
    tl, tr, br, bl = corners

    # Compute output dimensions
    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    width = int(max(width_top, width_bottom))

    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)
    height = int(max(height_left, height_right))

    # Destination points
    dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1],
    ], dtype=np.float32)

    # Perspective transform
    matrix = cv2.getPerspectiveTransform(corners, dst)
    warped = cv2.warpPerspective(image, matrix, (width, height))

    logger.info("Warped screen: %dx%d -> %dx%d", image.shape[1], image.shape[0], width, height)
    return warped


def ocr_warped_screen(warped: np.ndarray) -> str:
    """Run Tesseract OCR on the warped screen image.

    Upscales 3x and tries both light and dark background versions.

    Args:
        warped: Warped BGR image of the LCD screen.

    Returns:
        Best OCR text result.
    """
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

    # Upscale 3x for better OCR accuracy
    h, w = gray.shape
    upscaled = cv2.resize(gray, (w * 3, h * 3), interpolation=cv2.INTER_CUBIC)

    config = "--psm 6 --oem 3"
    results = []

    # Version 1: Direct (dark background, light text)
    text = pytesseract.image_to_string(upscaled, lang="eng", config=config).strip()
    results.append(("direct", text))

    # Version 2: Inverted (light background, dark text)
    inverted = cv2.bitwise_not(upscaled)
    text = pytesseract.image_to_string(inverted, lang="eng", config=config).strip()
    results.append(("inverted", text))

    # Version 3: Binary threshold (low)
    _, binary = cv2.threshold(upscaled, 50, 255, cv2.THRESH_BINARY)
    text = pytesseract.image_to_string(binary, lang="eng", config=config).strip()
    results.append(("binary_50", text))

    # Version 4: Binary threshold inverted
    binary_inv = cv2.bitwise_not(binary)
    text = pytesseract.image_to_string(binary_inv, lang="eng", config=config).strip()
    results.append(("binary_50_inv", text))

    # Version 5: Otsu
    _, otsu = cv2.threshold(upscaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    text = pytesseract.image_to_string(otsu, lang="eng", config=config).strip()
    results.append(("otsu", text))

    # Version 6: PSM 11 (sparse text) on binary
    text = pytesseract.image_to_string(binary, lang="eng", config="--psm 11 --oem 3").strip()
    results.append(("binary_psm11", text))

    # Select best result
    best_text = ""
    best_score = -1

    import re
    for name, text in results:
        text_lower = text.lower()
        label_count = sum(1 for lbl in ["sys", "dia", "pul"] if lbl in text_lower)
        digit_seqs = re.findall(r"\d{2,3}", text)
        digit_count = len(digit_seqs)
        score = label_count * 3 + digit_count * 2
        if label_count > 0 and digit_count > 0:
            score += 10
        logger.debug("Warp OCR '%s': score=%d, text=%s", name, score, repr(text[:60]))
        if score > best_score:
            best_score = score
            best_text = text

    # If no good result, return longest
    if best_score <= 0:
        results.sort(key=lambda r: len(r[1]), reverse=True)
        best_text = results[0][1] if results else ""

    return best_text


def detect_and_ocr(image_path: str, debug_output: bool = True) -> str:
    """Full pipeline: detect screen, warp, OCR.

    Args:
        image_path: Path to the input image.
        debug_output: If True, save warped image to output/debug_warped_*.png

    Returns:
        OCR text from the warped screen, or empty string if detection fails.
    """
    image = cv2.imread(image_path)
    if image is None:
        logger.error("Cannot read image: %s", image_path)
        return ""

    # Step 1: Detect screen
    corners = detect_screen(image)
    if corners is None:
        logger.warning("Screen detection failed for: %s", image_path)
        return ""

    # Step 2: Warp
    warped = warp_screen(image, corners)

    # Save debug output
    if debug_output:
        os.makedirs("output", exist_ok=True)
        filename = os.path.basename(image_path)
        debug_path = f"output/debug_warped_{filename}"
        cv2.imwrite(debug_path, warped)
        logger.info("Debug warped image saved: %s", debug_path)

    # Step 3: OCR
    text = ocr_warped_screen(warped)
    return text


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.DEBUG)

    if len(sys.argv) < 2:
        print("Usage: python screen_detector.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    text = detect_and_ocr(image_path)
    print(f"\nExtracted text:\n{text}")
