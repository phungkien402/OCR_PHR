"""Web interface for OCR Vital Signs pipeline.

Usage:
    uvicorn web_app:app --host 0.0.0.0 --port 8502
"""

import json
import os
import tempfile

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

app = FastAPI(title="OCR Vital Signs")

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>OCR Vital Signs</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: #f5f7fa; color: #333; min-height: 100vh; padding: 2rem;
}
.container { max-width: 900px; margin: 0 auto; }
h1 { text-align: center; margin-bottom: 0.5rem; color: #1a1a2e; font-size: 1.8rem; }
.subtitle { text-align: center; color: #666; margin-bottom: 2rem; font-size: 0.95rem; }
.tabs {
    display: flex; gap: 0; margin-bottom: 1.5rem; border-bottom: 2px solid #e0e0e0;
}
.tab {
    padding: 0.8rem 1.5rem; cursor: pointer; font-weight: 600; color: #888;
    border-bottom: 2px solid transparent; margin-bottom: -2px; transition: all 0.2s;
}
.tab:hover { color: #4a90d9; }
.tab.active { color: #4a90d9; border-bottom-color: #4a90d9; }
.tab-content { display: none; }
.tab-content.active { display: block; }
.card {
    background: #fff; border-radius: 12px; padding: 2rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.08); margin-bottom: 1.5rem;
}
.drop-zone {
    border: 2px dashed #ccc; border-radius: 8px; padding: 3rem 2rem;
    text-align: center; cursor: pointer; transition: all 0.2s; color: #888;
}
.drop-zone:hover, .drop-zone.dragover { border-color: #4a90d9; background: #f0f7ff; color: #4a90d9; }
.drop-zone p { margin-bottom: 0.5rem; font-size: 1.1rem; }
.drop-zone small { color: #aaa; }
.preview-container { margin-top: 1rem; text-align: center; display: none; }
.preview-container img { max-width: 100%; max-height: 400px; border-radius: 8px; border: 1px solid #eee; }
.filename { margin-top: 0.5rem; font-size: 0.85rem; color: #666; }
.btn {
    display: block; width: 100%; padding: 0.9rem; border: none; border-radius: 8px;
    font-size: 1rem; font-weight: 600; cursor: pointer; transition: all 0.2s; margin-top: 1.5rem;
}
.btn-primary { background: #4a90d9; color: #fff; }
.btn-primary:hover { background: #357abd; }
.btn-primary:disabled { background: #ccc; cursor: not-allowed; }
.spinner { display: none; text-align: center; padding: 2rem; }
.spinner::after {
    content: ''; display: inline-block; width: 40px; height: 40px;
    border: 4px solid #eee; border-top-color: #4a90d9; border-radius: 50%;
    animation: spin 0.8s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }
.result-section { display: none; }
.result-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem; }
.result-header h2 { font-size: 1.2rem; color: #1a1a2e; }
.vitals-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 1rem; margin-bottom: 1.5rem; }
.vital-card { background: #f8fafc; border-radius: 8px; padding: 1rem; text-align: center; border: 1px solid #e8ecf0; }
.vital-card.out-of-range { border-color: #fca5a5; background: #fef2f2; }
.vital-card .label-vn { font-size: 0.85rem; font-weight: 600; color: #1a1a2e; margin-bottom: 0.2rem; }
.vital-card .label-en { font-size: 0.7rem; color: #888; margin-bottom: 0.3rem; }
.vital-card .value { font-size: 1.5rem; font-weight: 700; color: #1a1a2e; }
.vital-card .value.null { color: #ccc; font-size: 1rem; }
.vital-card .value.out-of-range { color: #dc2626; }
.vital-card .unit { font-size: 0.7rem; color: #888; margin-top: 0.2rem; }
.vital-card .range { font-size: 0.65rem; color: #aaa; margin-top: 0.1rem; }
.json-output {
    background: #1e1e2e; color: #cdd6f4; border-radius: 8px; padding: 1.5rem;
    overflow-x: auto; font-family: 'JetBrains Mono', 'Fira Code', monospace;
    font-size: 0.85rem; line-height: 1.6; white-space: pre-wrap; word-break: break-word;
}
.json-output .key { color: #89b4fa; }
.json-output .string { color: #a6e3a1; }
.json-output .number { color: #fab387; }
.json-output .null { color: #6c7086; }
.json-output .bool { color: #cba6f7; }
.error-box {
    display: none; background: #fef2f2; border: 1px solid #fecaca;
    border-radius: 8px; padding: 1rem 1.5rem; color: #dc2626; margin-top: 1rem;
}
.raw-text-output {
    background: #1e1e2e; color: #a6e3a1; border-radius: 8px; padding: 1.5rem;
    font-family: 'JetBrains Mono', 'Fira Code', monospace; font-size: 0.9rem;
    line-height: 1.6; white-space: pre-wrap; word-break: break-word; min-height: 100px;
}
.mode-select {
    margin-top: 1rem; display: flex; gap: 0.5rem; align-items: center;
}
.mode-select label { font-weight: 600; color: #555; font-size: 0.9rem; }
.mode-select select {
    padding: 0.5rem 1rem; border: 1px solid #ddd; border-radius: 6px;
    font-size: 0.9rem; background: #fff; cursor: pointer;
}
.meta-badge {
    display: inline-block; background: #e8ecf0; border-radius: 4px;
    padding: 0.2rem 0.6rem; font-size: 0.75rem; color: #555; margin-top: 0.5rem;
}
input[type="file"] { display: none; }
</style>
</head>
<body>
<div class="container">
    <h1>OCR Vital Signs</h1>
    <p class="subtitle">Upload a blood pressure monitor image to extract vital signs</p>

    <div class="tabs">
        <div class="tab active" data-tab="process">Process</div>
        <div class="tab" data-tab="raw-ocr">Raw OCR Test</div>
    </div>

    <!-- TAB 1: Process -->
    <div class="tab-content active" id="tab-process">
        <div class="card">
            <div class="drop-zone" id="dropZone1">
                <p>Drop image here or click to browse</p>
                <small>Supports JPG, PNG</small>
            </div>
            <input type="file" id="fileInput1" accept="image/jpeg,image/png">
            <div class="preview-container" id="previewContainer1">
                <img id="previewImg1" alt="Preview">
                <p class="filename" id="fileName1"></p>
            </div>
            <button class="btn btn-primary" id="processBtn" disabled>Process</button>
        </div>
        <div class="spinner" id="spinner1"></div>
        <div class="error-box" id="errorBox1"></div>
        <div class="card result-section" id="resultSection1">
            <div class="result-header"><h2>Extracted Vitals</h2></div>
            <div class="vitals-grid" id="vitalsGrid"></div>
            <h2 style="font-size:1rem; margin-bottom:0.8rem; color:#1a1a2e;">Raw JSON</h2>
            <div class="json-output" id="jsonOutput"></div>
        </div>
    </div>

    <!-- TAB 2: Raw OCR Test -->
    <div class="tab-content" id="tab-raw-ocr">
        <div class="card">
            <div class="drop-zone" id="dropZone2">
                <p>Drop image here or click to browse</p>
                <small>Supports JPG, PNG</small>
            </div>
            <input type="file" id="fileInput2" accept="image/jpeg,image/png">
            <div class="preview-container" id="previewContainer2">
                <img id="previewImg2" alt="Preview">
                <p class="filename" id="fileName2"></p>
            </div>
            <div class="mode-select">
                <label>Model:</label>
                <select id="modelSelect">
                    <option value="qwen3_vl">Qwen3-VL:4b (vision)</option>
                    <option value="vietocr">VietOCR (handwritten)</option>
                </select>
            </div>
            <div class="prompt-field" id="promptField" style="margin-top:0.8rem;">
                <label style="font-weight:600; color:#555; font-size:0.9rem; display:block; margin-bottom:0.3rem;">Prompt (Qwen3-VL only):</label>
                <textarea id="promptInput" rows="3" style="width:100%; padding:0.6rem; border:1px solid #ddd; border-radius:6px; font-size:0.85rem; font-family:inherit; resize:vertical;">What text do you see in this image? List everything you can read.</textarea>
            </div>
            <button class="btn btn-primary" id="rawOcrBtn" disabled>Get Raw Model Output</button>
        </div>
        <div class="spinner" id="spinner2"></div>
        <div class="error-box" id="errorBox2"></div>
        <div class="card result-section" id="resultSection2">
            <div class="result-header"><h2>Raw model response (unfiltered)</h2></div>
            <div style="margin-bottom:0.8rem;">
                <span class="meta-badge" id="modelUsed"></span>
                <span class="meta-badge" id="promptUsed"></span>
            </div>
            <div class="raw-text-output" id="rawTextOutput" style="max-height:500px; overflow-y:auto;"></div>
        </div>
    </div>
</div>

<script>
// Tab switching
document.querySelectorAll('.tab').forEach(tab => {
    tab.addEventListener('click', () => {
        document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
        tab.classList.add('active');
        document.getElementById('tab-' + tab.dataset.tab).classList.add('active');
    });
});

// === TAB 1: Process ===
const dropZone1 = document.getElementById('dropZone1');
const fileInput1 = document.getElementById('fileInput1');
const previewContainer1 = document.getElementById('previewContainer1');
const previewImg1 = document.getElementById('previewImg1');
const fileName1 = document.getElementById('fileName1');
const processBtn = document.getElementById('processBtn');
const spinner1 = document.getElementById('spinner1');
const errorBox1 = document.getElementById('errorBox1');
const resultSection1 = document.getElementById('resultSection1');
let selectedFile1 = null;

dropZone1.addEventListener('click', () => fileInput1.click());
dropZone1.addEventListener('dragover', (e) => { e.preventDefault(); dropZone1.classList.add('dragover'); });
dropZone1.addEventListener('dragleave', () => dropZone1.classList.remove('dragover'));
dropZone1.addEventListener('drop', (e) => {
    e.preventDefault(); dropZone1.classList.remove('dragover');
    if (e.dataTransfer.files.length) handleFile1(e.dataTransfer.files[0]);
});
fileInput1.addEventListener('change', () => { if (fileInput1.files.length) handleFile1(fileInput1.files[0]); });

function handleFile1(file) {
    if (!file.type.match(/^image\/(jpeg|png)$/)) { showError1('Please upload a JPG or PNG image.'); return; }
    selectedFile1 = file;
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImg1.src = e.target.result;
        previewContainer1.style.display = 'block';
        fileName1.textContent = file.name + ' (' + (file.size / 1024).toFixed(1) + ' KB)';
    };
    reader.readAsDataURL(file);
    processBtn.disabled = false;
    resultSection1.style.display = 'none';
    errorBox1.style.display = 'none';
}

processBtn.addEventListener('click', async () => {
    if (!selectedFile1) return;
    processBtn.disabled = true;
    spinner1.style.display = 'block';
    resultSection1.style.display = 'none';
    errorBox1.style.display = 'none';
    const formData = new FormData();
    formData.append('file', selectedFile1);
    try {
        const resp = await fetch('/process', { method: 'POST', body: formData });
        const data = await resp.json();
        if (!resp.ok) { showError1(data.detail || 'Processing failed'); return; }
        displayResult(data);
    } catch (err) { showError1('Connection error: ' + err.message); }
    finally { spinner1.style.display = 'none'; processBtn.disabled = false; }
});

function showError1(msg) { errorBox1.textContent = msg; errorBox1.style.display = 'block'; }

function displayResult(data) {
    const vitals = data.vitals || {};
    const meta = data.fields_meta || {};
    const validation = data.validation || {};
    const bp = vitals.huyet_ap || {};

    const cards = [
        { key: 'huyet_ap.tam_thu', field: 'huyet_ap', sub: 'tam_thu', value: bp.tam_thu },
        { key: 'huyet_ap.tam_truong', field: 'huyet_ap', sub: 'tam_truong', value: bp.tam_truong },
        { key: 'mach', field: 'mach', value: vitals.mach },
        { key: 'nhiet_do', field: 'nhiet_do', value: vitals.nhiet_do },
        { key: 'spo2', field: 'spo2', value: vitals.spo2 },
        { key: 'nhip_tho', field: 'nhip_tho', value: vitals.nhip_tho },
        { key: 'can_nang', field: 'can_nang', value: vitals.can_nang },
        { key: 'chieu_cao', field: 'chieu_cao', value: vitals.chieu_cao },
    ];

    const vitalsGrid = document.getElementById('vitalsGrid');
    vitalsGrid.innerHTML = cards.map(c => {
        const info = meta[c.field] || {};
        const isNull = c.value == null;
        const isOutOfRange = validation[c.key] && validation[c.key].out_of_range;
        let labelVn = info.label_vn || c.field;
        let labelEn = info.label_en || '';
        let unit = info.unit || '';
        let normalRange = '';

        if (c.sub) {
            labelVn = c.sub === 'tam_thu' ? 'SYS (Tâm thu)' : 'DIA (Tâm trương)';
            labelEn = c.sub === 'tam_thu' ? 'Systolic' : 'Diastolic';
            unit = 'mmHg';
            const nr = info.normal_range;
            if (nr && nr[c.sub]) normalRange = nr[c.sub][0] + '-' + nr[c.sub][1];
        } else {
            const nr = info.normal_range;
            if (nr && !nr.tam_thu) normalRange = nr[0] + '-' + nr[1];
        }

        return '<div class="vital-card ' + (isOutOfRange ? 'out-of-range' : '') + '">' +
            '<div class="label-vn">' + labelVn + '</div>' +
            '<div class="label-en">' + labelEn + '</div>' +
            '<div class="value ' + (isNull ? 'null' : '') + (isOutOfRange ? ' out-of-range' : '') + '">' +
            (isNull ? '—' : c.value) + '</div>' +
            '<div class="unit">' + unit + '</div>' +
            (normalRange ? '<div class="range">Normal: ' + normalRange + '</div>' : '') +
            '</div>';
    }).join('');

    document.getElementById('jsonOutput').innerHTML = syntaxHighlight(JSON.stringify(data, null, 2));
    resultSection1.style.display = 'block';
}

// === TAB 2: Raw OCR Test ===
const dropZone2 = document.getElementById('dropZone2');
const fileInput2 = document.getElementById('fileInput2');
const previewContainer2 = document.getElementById('previewContainer2');
const previewImg2 = document.getElementById('previewImg2');
const fileName2 = document.getElementById('fileName2');
const rawOcrBtn = document.getElementById('rawOcrBtn');
const spinner2 = document.getElementById('spinner2');
const errorBox2 = document.getElementById('errorBox2');
const resultSection2 = document.getElementById('resultSection2');
const modelSelect = document.getElementById('modelSelect');
const promptField = document.getElementById('promptField');
const promptInput = document.getElementById('promptInput');
let selectedFile2 = null;

// Show/hide prompt field based on model selection
modelSelect.addEventListener('change', () => {
    promptField.style.display = modelSelect.value === 'qwen3_vl' ? 'block' : 'none';
});

dropZone2.addEventListener('click', () => fileInput2.click());
dropZone2.addEventListener('dragover', (e) => { e.preventDefault(); dropZone2.classList.add('dragover'); });
dropZone2.addEventListener('dragleave', () => dropZone2.classList.remove('dragover'));
dropZone2.addEventListener('drop', (e) => {
    e.preventDefault(); dropZone2.classList.remove('dragover');
    if (e.dataTransfer.files.length) handleFile2(e.dataTransfer.files[0]);
});
fileInput2.addEventListener('change', () => { if (fileInput2.files.length) handleFile2(fileInput2.files[0]); });

function handleFile2(file) {
    if (!file.type.match(/^image\/(jpeg|png)$/)) { showError2('Please upload a JPG or PNG image.'); return; }
    selectedFile2 = file;
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImg2.src = e.target.result;
        previewContainer2.style.display = 'block';
        fileName2.textContent = file.name + ' (' + (file.size / 1024).toFixed(1) + ' KB)';
    };
    reader.readAsDataURL(file);
    rawOcrBtn.disabled = false;
    resultSection2.style.display = 'none';
    errorBox2.style.display = 'none';
}

rawOcrBtn.addEventListener('click', async () => {
    if (!selectedFile2) return;
    rawOcrBtn.disabled = true;
    spinner2.style.display = 'block';
    resultSection2.style.display = 'none';
    errorBox2.style.display = 'none';
    const formData = new FormData();
    formData.append('file', selectedFile2);
    formData.append('model', modelSelect.value);
    formData.append('prompt', promptInput.value);
    try {
        const resp = await fetch('/raw-model-output', { method: 'POST', body: formData });
        const data = await resp.json();
        if (!resp.ok) { showError2(data.detail || 'Model request failed'); return; }
        displayRawOcr(data);
    } catch (err) { showError2('Connection error: ' + err.message); }
    finally { spinner2.style.display = 'none'; rawOcrBtn.disabled = false; }
});

function showError2(msg) { errorBox2.textContent = msg; errorBox2.style.display = 'block'; }

function displayRawOcr(data) {
    document.getElementById('modelUsed').textContent = 'Model: ' + data.model;
    document.getElementById('promptUsed').textContent = 'Prompt: ' + (data.prompt_used || '').substring(0, 60) + (data.prompt_used && data.prompt_used.length > 60 ? '...' : '');
    document.getElementById('rawTextOutput').textContent = data.raw_response || '(empty)';
    resultSection2.style.display = 'block';
}

// === Shared ===
function syntaxHighlight(json) {
    return json.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
        .replace(/("(\\u[a-zA-Z0-9]{4}|\\[^u]|[^\\"])*"(\s*:)?|\b(true|false)\b|-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?|\bnull\b)/g,
        function(match) {
            let cls = 'number';
            if (/^"/.test(match)) { cls = /:$/.test(match) ? 'key' : 'string'; }
            else if (/true|false/.test(match)) { cls = 'bool'; }
            else if (/null/.test(match)) { cls = 'null'; }
            return '<span class="' + cls + '">' + match + '</span>';
        });
}
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the frontend HTML page."""
    return HTML_PAGE


@app.post("/process")
async def process(file: UploadFile = File(...)):
    """Process an uploaded image through the full OCR vital signs pipeline."""
    if file.content_type not in ("image/jpeg", "image/png"):
        return JSONResponse(
            status_code=400,
            content={"detail": "Only JPG and PNG images are supported."},
        )

    suffix = ".png" if "png" in file.content_type else ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        from ocr_vitals.main import process_image

        result = process_image(tmp_path, device="cuda:0", mode="auto")
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"detail": f"Processing error: {str(e)}"},
        )
    finally:
        os.unlink(tmp_path)


@app.post("/raw-ocr")
async def raw_ocr(file: UploadFile = File(...), mode: str = Form("auto")):
    """Run only the OCR layer (no parsing, no validation).

    Returns raw text from the OCR engine.
    """
    if file.content_type not in ("image/jpeg", "image/png"):
        return JSONResponse(
            status_code=400,
            content={"detail": "Only JPG and PNG images are supported."},
        )

    if mode not in ("auto", "lcd", "handwritten"):
        mode = "auto"

    suffix = ".png" if "png" in file.content_type else ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        import cv2
        from ocr_vitals.ocr_engine import extract_text, detect_image_mode
        from ocr_vitals.preprocessor import preprocess_image_raw

        raw_img = preprocess_image_raw(tmp_path)

        if mode == "auto":
            detected_mode = detect_image_mode(raw_img)
        else:
            detected_mode = mode

        raw_text = extract_text(raw_img, device="cuda:0", mode=detected_mode,
                                image_path=tmp_path)

        if detected_mode == "lcd":
            engine = "qwen3_vl_4b_ollama"
        else:
            engine = "vietocr_vgg_transformer"

        return JSONResponse(content={
            "raw_text": raw_text,
            "mode_detected": detected_mode,
            "engine": engine,
        })
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"detail": f"OCR error: {str(e)}"},
        )
    finally:
        os.unlink(tmp_path)


@app.post("/raw-model-output")
async def raw_model_output(
    file: UploadFile = File(...),
    model: str = Form("qwen3_vl"),
    prompt: str = Form("What text do you see in this image? List everything you can read."),
):
    """Get the TRUE raw response from the model with zero processing.

    For qwen3_vl: sends image directly to Ollama with the given prompt.
    For vietocr: runs predictor.predict() on the full image.
    """
    if file.content_type not in ("image/jpeg", "image/png"):
        return JSONResponse(
            status_code=400,
            content={"detail": "Only JPG and PNG images are supported."},
        )

    if model not in ("qwen3_vl", "vietocr"):
        model = "qwen3_vl"

    suffix = ".png" if "png" in file.content_type else ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        if model == "qwen3_vl":
            import base64
            import cv2
            import requests

            img = cv2.imread(tmp_path)
            if img is None:
                return JSONResponse(
                    status_code=400,
                    content={"detail": "Failed to read image."},
                )

            success, img_encoded = cv2.imencode(".png", img)
            if not success:
                return JSONResponse(
                    status_code=500,
                    content={"detail": "Failed to encode image."},
                )

            img_b64 = base64.b64encode(img_encoded.tobytes()).decode()

            payload = {
                "model": "qwen3-vl:4b",
                "messages": [{
                    "role": "user",
                    "content": prompt,
                    "images": [img_b64],
                }],
                "stream": False,
            }

            resp = requests.post(
                "http://localhost:11434/api/chat",
                json=payload,
                timeout=120,
            )
            resp.raise_for_status()
            raw_response = resp.json()["message"]["content"]

            return JSONResponse(content={
                "raw_response": raw_response,
                "model": "qwen3-vl:4b (Ollama)",
                "prompt_used": prompt,
            })

        else:  # vietocr
            from PIL import Image as PILImage
            from ocr_vitals.ocr_engine import load_vietocr

            predictor = load_vietocr("cuda:0")
            pil_img = PILImage.open(tmp_path).convert("RGB")
            raw_response = predictor.predict(pil_img)

            return JSONResponse(content={
                "raw_response": raw_response,
                "model": "VietOCR vgg_transformer",
                "prompt_used": "(n/a — VietOCR has no prompt)",
            })

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"detail": f"Model error: {str(e)}"},
        )
    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8502)
