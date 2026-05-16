# OCR Vital Signs — Tóm tắt dự án

Đây là pipeline OCR để trích xuất các chỉ số sinh tồn (vital signs) từ ảnh y tế — chủ yếu là màn hình thiết bị (ví dụ: máy đo huyết áp, oximeter) và các biểu mẫu/nốt tay viết tay. Kết quả đầu ra là các file JSON có cấu trúc sẵn sàng cho downstream processing.

## Tổng quan

- Hỗ trợ 2 chế độ OCR chính:
  - `lcd` — đọc trực tiếp từ ảnh sản phẩm (màn hình LCD) bằng mô hình Vision-Language (Qwen3-VL qua Ollama)
  - `handwritten` — dùng VietOCR (mô hình `vgg_transformer`) để đọc chữ viết tay tiếng Việt
- Chế độ `auto` tự động xác định kiểu ảnh (LCD vs handwriting) và chọn engine phù hợp.
- Có fallback: nếu Ollama/Qwen không khả dụng thì dùng thao tác warp + Tesseract.
- Pipeline thực hiện: phát hiện kiểu ảnh → OCR → regex parser trích xuất trường (SYS/DIA/PUL...) → kiểm tra giới hạn sinh lý → xuất JSON.

## Cài đặt nhanh (Quickstart)

Yêu cầu tối thiểu:

- Python 3.10+
- Ollama (nếu muốn dùng Qwen3-VL) và model `qwen3-vl:4b`
- Tesseract OCR (dùng làm fallback)
- Thư viện Python: OpenCV, Pillow, PyTorch (nếu dùng VietOCR trên GPU), pytesseract, requests

Ví dụ cài nhanh:

```bash
# Ollama (nếu chưa cài)
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen3-vl:4b

# Python deps
pip install -r requirements.txt
# hoặc (dự phòng)
pip install vietocr easyocr opencv-python-headless pillow torch torchvision pytesseract requests
```

## Chạy nhanh (examples)

Chạy CLI cho 1 ảnh:

```bash
python main.py --input test_images/bp_monitor_lcd.png --output ./output/
```

Chạy cho cả thư mục:

```bash
python main.py --input test_images/ --output ./output/
```

Chọn chế độ OCR:

```bash
python main.py --input image.jpg --output ./output/ --mode lcd
python main.py --input image.jpg --output ./output/ --mode handwritten
python main.py --input image.jpg --output ./output/ --mode auto
```

Chạy web UI (FastAPI + Uvicorn):

```bash
uvicorn web_app:app --host 0.0.0.0 --port 8502
# rồi mở http://localhost:8502
```

## Đầu ra

Với mỗi file đầu vào sẽ có file JSON tương ứng trong `output/` chứa:
- `source_image`: tên file gốc
- `ocr_raw_text`: văn bản thô từ engine OCR
- `vitals`: object các trường (mach, nhiet_do, huyet_ap.{tam_thu,tam_truong}, nhip_tho, can_nang, chieu_cao, spo2)
- `missing_fields`: danh sách các trường không tìm thấy (giá trị null)
- `ocr_engine`: engine đã dùng (ví dụ `qwen3_vl_4b_ollama` hoặc `vietocr_vgg_transformer`)

Ví dụ cấu trúc JSON:

```json
{
  "source_image": "bp_monitor.jpg",
  "ocr_raw_text": "SYS: 128\nDIA: 78\nPUL: 72",
  "vitals": { "mach": 72, "huyet_ap": {"tam_thu":128, "tam_truong":78}, "nhiet_do": null },
  "missing_fields": ["nhiet_do","nhip_tho"],
  "ocr_engine": "qwen3_vl_4b_ollama"
}
```

## Cấu trúc chính của dự án

```
├── main.py                  # wrapper CLI duyệt từ project root
├── web_app.py               # FastAPI web UI
├── ocr_vitals/              # pipeline chính
│   ├── main.py              # hàm entry & xử lý thư mục/ảnh
│   ├── ocr_engine.py        # tích hợp Qwen3-VL, VietOCR, Tesseract
│   ├── preprocessor.py      # các phép xử lý ảnh (warp, threshold, v.v.)
│   ├── parser.py            # regex-based parser cho các trường vital
│   ├── validator.py         # kiểm tra khoảng giá trị sinh lý
│   └── config.py            # tham số cấu hình, device, thresholds
├── screen_detector.py       # phát hiện màn hình, perspective warp
├── benchmark_lcd.py         # script benchmark các engine OCR
├── test_images/             # ảnh mẫu
└── output/                  # nơi lưu JSON kết quả
```

## Lưu ý vận hành (Ops)

- Ollama expose API mặc định tại `http://localhost:11434` — đảm bảo Ollama đang chạy khi chọn `lcd` mode.
- VietOCR cố gắng khởi tạo trên GPU (`cuda:0`) nếu có; nếu thiếu VRAM sẽ fallback về CPU.
- Nếu gặp lỗi khi gọi model từ web UI, kiểm tra log và trạng thái Ollama/endpoint.

## Góp ý & Phát triển

- Để phát triển: tạo virtualenv, cài deps, chỉnh `ocr_vitals/config.py` theo môi trường GPU/CPU.
- Tests: `benchmark_lcd.py` và các script trong `ocr_vitals/tests/` (nếu có) giúp đánh giá.

---
_Nếu bạn muốn, mình có thể thêm phần hướng dẫn cài đặt chi tiết cho môi trường Ubuntu, script Docker Compose cho Ollama hoặc file `requirements.txt` cụ thể._ 

