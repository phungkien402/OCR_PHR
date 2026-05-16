# OCR Vital Signs Extraction Pipeline

Extract vital signs from medical images (blood pressure monitor displays and handwritten vital signs forms) using dual OCR engines and output structured JSON.

No LLM calls, no external APIs — pure OCR + rule-based parsing.

## Features

- **Dual OCR engine support:**
  - **EasyOCR** (English) for LCD/7-segment digital displays (blood pressure monitors)
  - **VietOCR** (`vgg_transformer`) for Vietnamese handwritten text (notebooks/forms)
- Auto-detection of image type (LCD vs handwritten)
- OpenCV preprocessing (CLAHE, denoising, adaptive thresholding)
- Regex-based parser for 7 vital sign fields
- LCD label recognition (SYS, DIA, PUL) with spatial proximity matching
- Physiological range validation
- CLI with single image and batch directory support
- Automatic GPU/CPU fallback

## Supported Vital Signs

| Field | Key | Type | Example |
|-------|-----|------|---------|
| Pulse | `mach` | integer | 100 |
| Temperature | `nhiet_do` | float | 37.0 |
| Blood Pressure | `huyet_ap` | object | `{"tam_thu": 110, "tam_truong": 65}` |
| Respiratory Rate | `nhip_tho` | integer | 18 |
| Weight | `can_nang` | float | 55.0 |
| Height | `chieu_cao` | integer | 160 |
| SpO2 | `spo2` | integer | 98 |

## Setup

### Requirements

- Python 3.10+
- CUDA 12.x (optional, falls back to CPU)

### Install

```bash
pip install vietocr easyocr opencv-python-headless pillow torch torchvision
```

## Usage

### Single image

```bash
python main.py --input /path/to/image.jpg --output ./output/
```

### Entire directory

```bash
python main.py --input /path/to/images/ --output ./output/
```

### Specify OCR mode

```bash
# LCD blood pressure monitor (uses EasyOCR)
python main.py --input bp_monitor.jpg --output ./output/ --mode lcd

# Handwritten Vietnamese text (uses VietOCR)
python main.py --input notebook.jpg --output ./output/ --mode handwritten

# Auto-detect (default)
python main.py --input image.jpg --output ./output/ --mode auto
```

### Specify GPU

```bash
python main.py --input image.jpg --output ./output/ --device cuda:1
```

### Verbose logging

```bash
python main.py --input image.jpg --output ./output/ --verbose
```

## Output Format

Each image produces one JSON file named `{original_filename}.json`:

```json
{
  "source_image": "bp_monitor.jpg",
  "ocr_raw_text": "SYS 128\nDIA 78\nPUL 72\nmmHg",
  "vitals": {
    "mach": 72,
    "nhiet_do": null,
    "huyet_ap": {
      "tam_thu": 128,
      "tam_truong": 78
    },
    "nhip_tho": null,
    "can_nang": null,
    "chieu_cao": null,
    "spo2": null
  },
  "validation": {},
  "missing_fields": ["nhiet_do", "nhip_tho", "can_nang", "chieu_cao", "spo2"],
  "ocr_engine": "easyocr_en",
  "processed_at": "2026-05-16T11:25:00",
  "units_detected": ["mmhg"]
}
```

### Validation

Fields outside physiological ranges are flagged:

```json
"validation": {
  "spo2": {
    "out_of_range": true,
    "value": 40,
    "expected": "50-100"
  }
}
```

### Missing Fields

Fields not found in the image are set to `null` and listed:

```json
"missing_fields": ["nhip_tho", "can_nang"]
```

### Error Handling

If OCR fails on an image, the JSON contains an error field instead of crashing:

```json
{
  "source_image": "corrupted.jpg",
  "error": "Cannot read image: corrupted.jpg",
  "ocr_engine": "unknown",
  "processed_at": "2026-05-16T11:25:00"
}
```

## OCR Mode Selection

| Mode | Engine | Best For |
|------|--------|----------|
| `lcd` | EasyOCR (English) | Blood pressure monitors, pulse oximeters, digital thermometers |
| `handwritten` | VietOCR (Vietnamese) | Handwritten vital signs notebooks, medical forms |
| `auto` | Auto-detect | Mixed input — detects based on contrast and region analysis |

### LCD Label Recognition

For blood pressure monitors, the parser recognizes English labels:
- `SYS` → systolic blood pressure (`huyet_ap.tam_thu`)
- `DIA` → diastolic blood pressure (`huyet_ap.tam_truong`)
- `PUL` / `PUL/min` → pulse rate (`mach`)
- `mmHg`, `kPa` → unit metadata (stored but doesn't affect values)

## Project Structure

```
ocr_vitals/
├── main.py              # Pipeline logic and CLI
├── ocr_engine.py        # Dual OCR engine (VietOCR + EasyOCR)
├── preprocessor.py      # OpenCV image preprocessing
├── parser.py            # Regex-based vital signs parser + LCD labels
├── validator.py         # Physiological range validation
├── config.py            # GPU device, thresholds, field keywords
├── test_images/         # Put test images here
└── output/              # JSON outputs go here
```

## GPU Notes

- Both engines attempt to load on the configured GPU (default: `cuda:0`)
- If VRAM is insufficient, they automatically fall back to CPU with a warning
- Use `--device cpu` to force CPU inference
