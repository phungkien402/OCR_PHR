# Task: Build OCR Vital Signs Extraction Pipeline

## Objective
Build a Python pipeline that extracts vital signs from medical images (handheld blood pressure monitor displays and handwritten vital signs notebooks/forms) and outputs structured JSON files.

No LLM calls, no external APIs. Pure OCR + rule-based parsing only.

---

## Server Environment
- OS: Ubuntu/Debian
- GPU: 2x Tesla V100 16GB (each GPU has ~6GB VRAM free — server is running another production workload)
- RAM: 128GB
- CUDA: 12.8
- Python: 3.10+

---

## Tech Stack

| Component | Library |
|---|---|
| OCR engine | **VietOCR** (`vgg_transformer` model) |
| Image preprocessing | **OpenCV** |
| Parsing | **regex + rule-based** |
| Output | Python stdlib `json` |

Install dependencies:
```bash
pip install vietocr opencv-python-headless pillow torch torchvision
```

Do NOT install paddleocr, easyocr, or tesseract — keep it minimal.

---

## Project Structure

```
ocr_vitals/
├── main.py              # Entry point
├── ocr_engine.py        # VietOCR loader and text extractor
├── preprocessor.py      # OpenCV image preprocessing
├── parser.py            # Regex-based vital signs parser
├── validator.py         # Physiological range validation
├── config.py            # GPU device, thresholds, field keywords
├── test_images/         # Put test images here
└── output/              # JSON outputs go here
```

---

## Implementation Details

### config.py
```python
DEVICE = "cuda:0"  # fallback to "cpu" if VRAM insufficient

FIELD_KEYWORDS = {
    "mach":      ["mạch", "pulse", "hr", "heart rate", "p:"],
    "nhiet_do":  ["nhiệt độ", "nhiệt", "temp", "t:", "°c"],
    "huyet_ap":  ["huyết áp", "ha", "bp", "blood pressure"],
    "nhip_tho":  ["nhịp thở", "rr", "nhịp"],
    "can_nang":  ["cân nặng", "cân", "weight", "kg"],
    "chieu_cao": ["chiều cao", "cao", "height", "cm"],
    "spo2":      ["spo2", "spо2", "o2", "oxy"],
}

RANGES = {
    "mach":      (30, 200),
    "nhiet_do":  (34.0, 42.0),
    "huyet_ap":  {"tam_thu": (60, 250), "tam_truong": (30, 150)},
    "nhip_tho":  (5, 60),
    "can_nang":  (1, 300),
    "chieu_cao": (30, 250),
    "spo2":      (50, 100),
}
```

### preprocessor.py
- Convert to grayscale
- Apply CLAHE for contrast enhancement
- Denoise with `fastNlMeansDenoising`
- Resize if shortest dimension < 1000px (preserve aspect ratio)
- Apply Otsu thresholding for LCD/digital display images
- Apply adaptive thresholding for handwritten images
- Return preprocessed image as numpy array

### ocr_engine.py
- Load VietOCR `vgg_transformer` model on the configured device
- Accept preprocessed image, return raw extracted text string
- Handle multi-region images: detect bounding boxes, OCR each region, concatenate results

### parser.py
Extract the following fields from raw OCR text:

- `mach` → integer (e.g. 100)
- `nhiet_do` → float (e.g. 37.0)
- `huyet_ap` → object with `tam_thu` (systolic) and `tam_truong` (diastolic), parsed from pattern `NNN/NN`
- `nhip_tho` → integer
- `can_nang` → float
- `chieu_cao` → integer
- `spo2` → integer

Logic: for each field, search for its keyword in the text (case-insensitive, diacritic-tolerant), then extract the nearest number. For `huyet_ap`, find pattern `\d{2,3}/\d{2,3}`.

### validator.py
- For each extracted field, check if value is within the physiological range defined in `config.py`
- If out of range → add to `validation` dict with `out_of_range: true`, actual value, and expected range string
- Fields not found → add to `missing_fields` list

### main.py
CLI interface:
```bash
# Single image
python main.py --input /path/to/image.jpg --output ./output/

# Entire directory
python main.py --input /path/to/images/ --output ./output/

# Specify GPU
python main.py --input image.jpg --output ./output/ --device cuda:1
```

Each image produces one JSON file named `{original_filename}.json` in the output directory.

---

## Output JSON Schema

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
    "nhip_tho": 50,
    "can_nang": 55.0,
    "chieu_cao": 160,
    "spo2": 40
  },
  "validation": {
    "spo2": {
      "out_of_range": true,
      "value": 40,
      "expected": "50-100"
    }
  },
  "missing_fields": [],
  "ocr_engine": "vietocr_vgg_transformer",
  "processed_at": "2026-05-16T11:25:00"
}
```

Fields not found in the image → set to `null` and add field name to `missing_fields`.

---

## Additional Requirements

1. Add a `README.md` with setup instructions, usage examples, and output format explanation
2. Add a `.gitignore` ignoring: `__pycache__/`, `*.pyc`, `output/`, `test_images/`, `.venv/`, `*.pth` (model weights)
3. If VRAM is insufficient when loading VietOCR → automatically fall back to CPU with a warning log
4. Use Python `logging` module (not print statements) for all status/error messages
5. The pipeline must handle errors gracefully: if OCR fails on an image, write a JSON with `"error": "<reason>"` instead of crashing

---

## Git Instructions

After implementing and verifying the code runs without errors:

1. Initialize git repo (if not already):
   ```bash
   git init
   git remote add origin https://github.com/phungkien402/OCR_PHR.git
   ```

2. Stage and commit:
   ```bash
   git add .
   git commit -m "feat: initial OCR vital signs extraction pipeline

   - VietOCR vgg_transformer for Vietnamese text (print + handwritten)
   - OpenCV preprocessing (CLAHE, denoise, threshold)
   - Regex-based parser for 7 vital sign fields
   - Physiological range validator
   - CLI interface with single image and batch directory support
   - JSON output per image"
   ```

3. Push to main branch:
   ```bash
   git branch -M main
   git push -u origin main
   ```

---

## Verification Before Push

Run a quick sanity check before committing:
```bash
python main.py --input test_images/ --output output/
```

Confirm:
- No Python exceptions
- At least one JSON file produced in `output/`
- JSON contains the expected fields structure
- `missing_fields` and `validation` keys are present (even if empty)
