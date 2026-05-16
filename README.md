# OCR Vital Signs Extraction Pipeline

Extract vital signs from medical images (blood pressure monitor displays and handwritten vital signs forms) using a vision-language model (Qwen3-VL) and VietOCR, outputting structured JSON.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Input Image                                │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│              Image Mode Detection                             │
│   (contrast ratio, histogram bimodality, region analysis)    │
└──────────┬──────────────────────────────────┬───────────────┘
           │                                  │
     LCD detected                     Handwritten detected
           │                                  │
           ▼                                  ▼
┌─────────────────────────┐    ┌──────────────────────────────┐
│  Screen Detector         │    │  VietOCR (vgg_transformer)    │
│  (OpenCV edge/contour    │    │  Vietnamese handwritten text  │
│   → perspective warp)    │    └──────────────────────────────┘
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│  Qwen3-VL:2b (Ollama)   │ ◄── Primary LCD engine
│  Vision-language model   │
│  localhost:11434         │
└──────────┬──────────────┘
           │ (fallback if Ollama unavailable)
           ▼
┌─────────────────────────┐
│  Tesseract OCR           │ ◄── Fallback LCD engine
│  Multi-pass threshold    │
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────┐
│              Regex Parser + Vital Signs Extraction            │
│   SYS → huyet_ap.tam_thu, DIA → huyet_ap.tam_truong         │
│   PUL → mach                                                 │
└──────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────┐
│              Physiological Range Validation                   │
└──────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────┐
│              Structured JSON Output                           │
└─────────────────────────────────────────────────────────────┘
```

## Features

- **LCD OCR pipeline:**
  - Screen detection + perspective warp (OpenCV)
  - **Qwen3-VL:2b** via Ollama — vision-language model that reads LCD digits directly
  - Fallback to Tesseract if Ollama is unavailable
- **Handwritten OCR:** VietOCR (`vgg_transformer`) for Vietnamese handwritten text
- Auto-detection of image type (LCD vs handwritten)
- Regex-based parser for 7 vital sign fields
- LCD label recognition (SYS, DIA, PUL) with structured output
- Physiological range validation
- CLI with single image and batch directory support
- Automatic GPU/CPU fallback

## Prerequisites

- Python 3.10+
- **Ollama** with `qwen3-vl:2b` model (for LCD mode)
- Tesseract OCR (fallback)
- CUDA 12.x (optional, falls back to CPU)

### Install Ollama + Qwen3-VL

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull the vision-language model
ollama pull qwen3-vl:2b

# Verify it's running
curl http://localhost:11434/api/tags
```

### Install Python dependencies

```bash
pip install vietocr easyocr opencv-python-headless pillow torch torchvision pytesseract requests
```

## Supported Vital Signs

| Field | Key | Type | Example |
|-------|-----|------|---------|
| Pulse | `mach` | integer | 72 |
| Temperature | `nhiet_do` | float | 37.0 |
| Blood Pressure | `huyet_ap` | object | `{"tam_thu": 128, "tam_truong": 78}` |
| Respiratory Rate | `nhip_tho` | integer | 18 |
| Weight | `can_nang` | float | 55.0 |
| Height | `chieu_cao` | integer | 160 |
| SpO2 | `spo2` | integer | 98 |

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
# LCD blood pressure monitor (Qwen3-VL + screen warp)
python main.py --input bp_monitor.jpg --output ./output/ --mode lcd

# Handwritten Vietnamese text (VietOCR)
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
  "ocr_raw_text": "SYS: 128\nDIA: 78\nPUL: 72",
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
  "ocr_engine": "qwen3_vl_2b_ollama",
  "processed_at": "2026-05-16T16:46:36",
  "units_detected": []
}
```

### Validation

Fields outside physiological ranges are flagged:

```json
"validation": {
  "mach": {
    "out_of_range": true,
    "value": 13,
    "expected": "30-200"
  }
}
```

### Missing Fields

Fields not found in the image are set to `null` and listed:

```json
"missing_fields": ["nhip_tho", "can_nang"]
```

## OCR Mode Selection

| Mode | Primary Engine | Fallback | Best For |
|------|---------------|----------|----------|
| `lcd` | Qwen3-VL:2b (Ollama) | Tesseract | Blood pressure monitors, pulse oximeters, digital thermometers |
| `handwritten` | VietOCR (Vietnamese) | — | Handwritten vital signs notebooks, medical forms |
| `auto` | Auto-detect | — | Mixed input — detects based on contrast and region analysis |

### LCD Pipeline Detail

1. **Screen Detection** — OpenCV edge detection + contour analysis finds the LCD rectangle
2. **Perspective Warp** — Corrects viewing angle to get a flat front-on view
3. **Qwen3-VL:2b** — Vision-language model reads digits directly from the warped image
4. **Fallback** — If Ollama is unavailable, Tesseract runs multi-pass OCR on the warped image

### LCD Label Recognition

For blood pressure monitors, the parser recognizes:
- `SYS` → systolic blood pressure (`huyet_ap.tam_thu`)
- `DIA` → diastolic blood pressure (`huyet_ap.tam_truong`)
- `PUL` / `PUL/min` → pulse rate (`mach`)

## Benchmark Results

Tested on real product photo (D2-65A blood pressure monitor):

| Engine | SYS | DIA | PUL | Score |
|--------|-----|-----|-----|-------|
| Tesseract (all modes) | ✗ | ✗ | ✗ | 0/3 |
| EasyOCR | ✗ | ✗ | ✗ | 0/3 |
| TrOCR | ✗ | ✗ | ✗ | 0/3 |
| **Qwen3-VL:2b (original)** | ✓ | — | — | 1/3 |
| **Qwen3-VL:2b (warped)** | ✓ | ✓ | ~✓ | 2/3 |

Qwen3-VL is the only engine capable of reading real-world LCD displays with low contrast.

## Project Structure

```
├── main.py                  # CLI entry point
├── ocr_vitals/
│   ├── main.py              # Pipeline logic
│   ├── ocr_engine.py        # Qwen3-VL + Tesseract + VietOCR engines
│   ├── preprocessor.py      # OpenCV image preprocessing
│   ├── parser.py            # Regex-based vital signs parser
│   ├── validator.py         # Physiological range validation
│   └── config.py            # GPU device, thresholds, field keywords
├── screen_detector.py       # LCD screen detection + perspective warp
├── benchmark_lcd.py         # Multi-engine OCR benchmark
├── test_images/             # Test images
└── output/                  # JSON outputs + debug images
```

## GPU Notes

- Qwen3-VL runs via Ollama (manages its own GPU/CPU allocation)
- VietOCR attempts to load on the configured GPU (default: `cuda:0`)
- If VRAM is insufficient, VietOCR automatically falls back to CPU
- Use `--device cpu` to force CPU inference for VietOCR
