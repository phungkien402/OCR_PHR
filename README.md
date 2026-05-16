# OCR Vital Signs Extraction Pipeline

Trích xuất **dấu hiệu sinh tồn** từ ảnh y khoa (màn hình máy đo huyết áp, ảnh biểu mẫu/ghi chú viết tay) bằng **Vision-Language Model (Qwen3-VL qua Ollama)** và **VietOCR**, sau đó chuẩn hoá thành **JSON có cấu trúc**.

## Kiến trúc tổng quan

```
┌─────────────────────────────────────────────────────────────┐
│                        Input Image                          │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                 Image Mode Detection                         │
│ (contrast ratio, histogram bimodality, region analysis, ...) │
└──────────┬──────────────────────────────────┬───────────────┘
           │                                  │
     LCD detected                      Handwritten detected
           │                                  │
           ▼                                  ▼
┌─────────────────────────┐      ┌──────────────────────────────┐
│  Qwen3-VL:4b (Ollama)    │      │  VietOCR (vgg_transformer)    │
│  Read LCD digits directly │      │  Vietnamese handwritten text  │
│  localhost:11434          │      └──────────────────────────────┘
└──────────┬───────────────┘
           │ (fallback if Ollama unavailable)
           ▼
┌─────────────────────────┐
│  Screen Warp + Tesseract │  ◄── Fallback only
│  Perspective correction  │
│  + multi-pass OCR        │
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────┐
│             Regex Parser + Vital Signs Extraction            │
│ SYS → huyet_ap.tam_thu | DIA → huyet_ap.tam_truong | PUL → mach │
└─────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────┐
│            Physiological Range Validation                    │
└─────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────┐
│              Structured JSON Output                          │
└─────────────────────────────────────────────────────────────┘
```

## Tính năng

- **LCD OCR pipeline**
  - **Qwen3-VL:4b** qua **Ollama** đọc trực tiếp chữ số trên màn hình LCD từ ảnh gốc (không cần warp)
  - Fallback sang **screen warp + Tesseract** khi Ollama không khả dụng
- **Handwritten OCR**: **VietOCR** (`vgg_transformer`) cho chữ viết tay tiếng Việt
- Tự động phát hiện loại ảnh (**LCD** vs **handwritten**)
- Parser dựa trên regex cho **7 trường** dấu hiệu sinh tồn
- Nhận dạng nhãn LCD (SYS/DIA/PUL) và xuất ra output có cấu trúc
- Kiểm tra **ngưỡng sinh lý** (range validation)
- CLI hỗ trợ chạy **1 ảnh** hoặc **cả thư mục**
- Tự fallback **GPU/CPU**

## Yêu cầu

- Python **3.10+**
- **Ollama** + model `qwen3-vl:4b` (cho chế độ LCD)
- Tesseract OCR (fallback)
- CUDA 12.x (tuỳ chọn, sẽ tự fallback CPU nếu không có)

### Cài Ollama + Qwen3-VL

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull the vision-language model
ollama pull qwen3-vl:4b

# Verify it's running
curl http://localhost:11434/api/tags
```

### Cài dependencies Python

```bash
pip install vietocr easyocr opencv-python-headless pillow torch torchvision pytesseract requests
```

## Các trường dấu hiệu sinh tồn hỗ trợ

| Field | Key | Type | Example |
|-------|-----|------|---------|
| Pulse | `mach` | integer | 72 |
| Temperature | `nhiet_do` | float | 37.0 |
| Blood Pressure | `huyet_ap` | object | `{"tam_thu": 128, "tam_truong": 78}` |
| Respiratory Rate | `nhip_tho` | integer | 18 |
| Weight | `can_nang` | float | 55.0 |
| Height | `chieu_cao` | integer | 160 |
| SpO2 | `spo2` | integer | 98 |

## Cách dùng

### Chạy 1 ảnh

```bash
python main.py --input /path/to/image.jpg --output ./output/
```

### Chạy cả thư mục

```bash
python main.py --input /path/to/images/ --output ./output/
```

### Chỉ định chế độ OCR

```bash
# LCD blood pressure monitor (Qwen3-VL + screen warp fallback)
python main.py --input bp_monitor.jpg --output ./output/ --mode lcd

# Handwritten Vietnamese text (VietOCR)
python main.py --input notebook.jpg --output ./output/ --mode handwritten

# Auto-detect (default)
python main.py --input image.jpg --output ./output/ --mode auto
```

### Chỉ định GPU

```bash
python main.py --input image.jpg --output ./output/ --device cuda:1
```

### Bật verbose log

```bash
python main.py --input image.jpg --output ./output/ --verbose
```

## Định dạng output

Mỗi ảnh tạo ra 1 file JSON tên `{original_filename}.json`:

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

Các field ngoài ngưỡng sinh lý sẽ bị gắn cờ:

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

Những trường không tìm thấy sẽ được đặt `null` và liệt kê:

```json
"missing_fields": ["nhip_tho", "can_nang"]
```

## Chọn OCR mode

| Mode | Primary Engine | Fallback | Best For |
|------|----------------|----------|----------|
| `lcd` | Qwen3-VL:4b (Ollama) | Screen warp + Tesseract | Máy đo huyết áp, SpO2, nhiệt kế điện tử |
| `handwritten` | VietOCR (Vietnamese) | — | Sổ ghi dấu hiệu sinh tồn, phiếu khám, form viết tay |
| `auto` | Auto-detect | — | Input hỗn hợp |

### LCD pipeline detail

1. **Qwen3-VL:4b** đọc chữ số trực tiếp từ ảnh gốc (không cần preprocessing)
2. **Fallback**: nếu Ollama không chạy → screen warp + Tesseract multi-pass OCR

### Nhận dạng nhãn LCD

Parser nhận dạng:

- `SYS` → huyết áp tâm thu (`huyet_ap.tam_thu`)
- `DIA` → huyết áp tâm trương (`huyet_ap.tam_truong`)
- `PUL` / `PUL/min` → mạch (`mach`)

## Benchmark

Test trên ảnh thật (D2-65A blood pressure monitor):

| Engine | SYS | DIA | PUL | Score | Time |
|--------|-----|-----|-----|-------|------|
| Tesseract (all modes) | ✗ | ✗ | ✗ | 0/3 | ~2s |
| EasyOCR | ✗ | ✗ | ✗ | 0/3 | ~4s |
| Qwen3-VL:2b (original) | ✓ | ✗ | ✗ | 1/3 | ~5s |
| Qwen3-VL:2b (warped) | ✓ | ✓ | ~✓ | 2/3 | ~1s |
| **Qwen3-VL:4b (original)** | **✓** | **✓** | **✓** | **3/3** | **7s** |
| Qwen3-VL:4b (warped) | ✓ | ✓ | ~✓ | 2/3 | ~8s |

Nhận xét: Qwen3-VL:4b trên ảnh gốc là engine duy nhất đạt điểm tuyệt đối; warp đôi khi làm giảm độ chính xác.

## Cấu trúc project

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

## Ghi chú GPU

- Qwen3-VL chạy qua Ollama (Ollama tự quản lý GPU/CPU)
- VietOCR cố gắng load theo GPU cấu hình (mặc định `cuda:0`)
- Nếu thiếu VRAM, VietOCR sẽ tự fallback CPU
- Dùng `--device cpu` để ép VietOCR chạy CPU
