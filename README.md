# OCR Vital Signs Extraction Pipeline

Extract vital signs from medical images (blood pressure monitor displays and handwritten vital signs forms) using VietOCR and output structured JSON.

No LLM calls, no external APIs — pure OCR + rule-based parsing.

## Features

- Vietnamese text recognition (printed + handwritten) via VietOCR `vgg_transformer`
- OpenCV preprocessing (CLAHE, denoising, adaptive thresholding)
- Regex-based parser for 7 vital sign fields
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
pip install vietocr opencv-python-headless pillow torch torchvision
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
  "source_image": "benh_nhan_01.jpg",
  "ocr_raw_text": "Mạch: 100\nNhiệt độ: 37\nHuyết áp: 110/65\n...",
  "vitals": {
    "mach": 100,
    "nhiet_do": 37.0,
    "huyet_ap": {
      "tam_thu": 110,
      "tam_truong": 65
    },
    "nhip_tho": 18,
    "can_nang": 55.0,
    "chieu_cao": 160,
    "spo2": 98
  },
  "validation": {},
  "missing_fields": [],
  "ocr_engine": "vietocr_vgg_transformer",
  "processed_at": "2026-05-16T11:25:00"
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
  "ocr_engine": "vietocr_vgg_transformer",
  "processed_at": "2026-05-16T11:25:00"
}
```

## Project Structure

```
ocr_vitals/
├── main.py              # Pipeline logic and CLI
├── ocr_engine.py        # VietOCR loader and text extractor
├── preprocessor.py      # OpenCV image preprocessing
├── parser.py            # Regex-based vital signs parser
├── validator.py         # Physiological range validation
├── config.py            # GPU device, thresholds, field keywords
├── test_images/         # Put test images here
└── output/              # JSON outputs go here
```

## GPU Notes

- The pipeline attempts to load the model on the configured GPU (default: `cuda:0`)
- If VRAM is insufficient, it automatically falls back to CPU with a warning
- Use `--device cpu` to force CPU inference
