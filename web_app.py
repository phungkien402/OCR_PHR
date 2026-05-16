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
    background: #f5f7fa; color: #333; min-height: 100vh; padding: 1.5rem;
}
.container { max-width: 1200px; margin: 0 auto; }
h1 { text-align: center; margin-bottom: 0.3rem; color: #1a1a2e; font-size: 1.6rem; }
.subtitle { text-align: center; color: #666; margin-bottom: 1.5rem; font-size: 0.9rem; }
.tabs { display: flex; gap: 0; margin-bottom: 1.2rem; border-bottom: 2px solid #e0e0e0; }
.tab {
    padding: 0.7rem 1.3rem; cursor: pointer; font-weight: 600; color: #888;
    border-bottom: 2px solid transparent; margin-bottom: -2px; transition: all 0.2s;
}
.tab:hover { color: #4a90d9; }
.tab.active { color: #4a90d9; border-bottom-color: #4a90d9; }
.tab-content { display: none; }
.tab-content.active { display: block; }
.two-col { display: grid; grid-template-columns: 1.5fr 1fr; gap: 1.2rem; }
@media (max-width: 768px) { .two-col { grid-template-columns: 1fr; } }
.panel {
    background: #fff; border-radius: 10px; padding: 1.5rem;
    box-shadow: 0 2px 10px rgba(0,0,0,0.06);
}
.panel-dark {
    background: #1e1e2e; border-radius: 10px; padding: 1.2rem;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1); display: flex; flex-direction: column;
}
.panel-dark-header {
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 0.8rem;
}
.panel-dark-header h3 { color: #cdd6f4; font-size: 0.9rem; font-weight: 600; }
.drop-zone {
    border: 2px dashed #ccc; border-radius: 8px; padding: 2.5rem 1.5rem;
    text-align: center; cursor: pointer; transition: all 0.2s; color: #888;
}
.drop-zone:hover, .drop-zone.dragover { border-color: #4a90d9; background: #f0f7ff; color: #4a90d9; }
.drop-zone p { margin-bottom: 0.3rem; font-size: 1rem; }
.drop-zone small { color: #aaa; font-size: 0.8rem; }
.preview-container { margin-top: 1rem; text-align: center; display: none; }
.preview-container img { max-width: 100%; max-height: 500px; border-radius: 8px; border: 1px solid #eee; }
.filename { margin-top: 0.4rem; font-size: 0.8rem; color: #666; }
.vitals-table { width: 100%; border-collapse: collapse; margin-top: 1rem; font-size: 0.85rem; }
.vitals-table th {
    text-align: left; padding: 0.5rem 0.6rem; border-bottom: 2px solid #e0e0e0;
    color: #555; font-weight: 600; font-size: 0.8rem;
}
.vitals-table td { padding: 0.5rem 0.6rem; border-bottom: 1px solid #f0f0f0; }
.vitals-table tr:last-child td { border-bottom: none; }
.vitals-table .val { font-weight: 600; color: #1a1a2e; }
.vitals-table .null { color: #ccc; }
.btn {
    display: block; width: 100%; padding: 0.85rem; border: none; border-radius: 8px;
    font-size: 1rem; font-weight: 600; cursor: pointer; transition: all 0.2s; margin-top: 1.2rem;
}
.btn-primary { background: #4a90d9; color: #fff; }
.btn-primary:hover { background: #357abd; }
.btn-primary:disabled { background: #ccc; cursor: not-allowed; }
.btn-sm {
    display: inline-block; width: auto; padding: 0.35rem 0.8rem; margin: 0;
    font-size: 0.75rem; border-radius: 5px; background: #4a90d9; color: #fff;
    border: none; cursor: pointer;
}
.btn-sm:hover { background: #357abd; }
.status-bar {
    display: flex; gap: 1.5rem; align-items: center; justify-content: center;
    margin-top: 1rem; font-size: 0.82rem; color: #666;
}
.status-bar span strong { color: #1a1a2e; }
.spinner { display: none; text-align: center; padding: 1.5rem; }
.spinner::after {
    content: ''; display: inline-block; width: 36px; height: 36px;
    border: 4px solid #eee; border-top-color: #4a90d9; border-radius: 50%;
    animation: spin 0.8s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }
.error-box {
    display: none; background: #fef2f2; border: 1px solid #fecaca;
    border-radius: 8px; padding: 0.8rem 1.2rem; color: #dc2626; margin-top: 1rem; font-size: 0.85rem;
}
.json-output {
    flex: 1; overflow-y: auto; color: #cdd6f4;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    font-size: 0.78rem; line-height: 1.5; white-space: pre-wrap; word-break: break-word;
}
.json-output .key { color: #89b4fa; }
.json-output .string { color: #a6e3a1; }
.json-output .number { color: #fab387; }
.json-output .null { color: #6c7086; }
.json-output .bool { color: #cba6f7; }
.raw-text-output {
    background: #1e1e2e; color: #a6e3a1; border-radius: 8px; padding: 1.2rem;
    font-family: 'JetBrains Mono', 'Fira Code', monospace; font-size: 0.85rem;
    line-height: 1.5; white-space: pre-wrap; word-break: break-word;
    min-height: 100px; max-height: 500px; overflow-y: auto;
}
.mode-select { margin-top: 0.8rem; display: flex; gap: 0.5rem; align-items: center; }
.mode-select label { font-weight: 600; color: #555; font-size: 0.85rem; }
.mode-select select {
    padding: 0.4rem 0.8rem; border: 1px solid #ddd; border-radius: 6px;
    font-size: 0.85rem; background: #fff; cursor: pointer;
}
.meta-badge {
    display: inline-block; background: #e8ecf0; border-radius: 4px;
    padding: 0.2rem 0.5rem; font-size: 0.72rem; color: #555; margin-right: 0.4rem;
}
input[type="file"] { display: none; }
.placeholder-text { color: #6c7086; font-size: 0.8rem; text-align: center; padding: 3rem 1rem; }
</style>
</head>
<body>
<div class="container">
    <h1>OCR Vital Signs</h1>
    <p class="subtitle">Extract vital signs from medical images using Qwen3-VL + VietOCR</p>

    <div class="tabs">
        <div class="tab active" data-tab="process">Process</div>
        <div class="tab" data-tab="raw-ocr">Raw OCR Test</div>
    </div>

    <!-- TAB 1: Process -->
    <div class="tab-content active" id="tab-process">
        <div class="two-col">
            <!-- Left column -->
            <div>
                <div class="panel">
                    <div class="drop-zone" id="dropZone1">
                        <p>Drop image here or click to browse</p>
                        <small>JPG, PNG supported</small>
                    </div>
                    <input type="file" id="fileInput1" accept="image/jpeg,image/png">
                    <div class="preview-container" id="previewContainer1">
                        <img id="previewImg1" alt="Preview">
                        <p class="filename" id="fileName1"></p>
                    </div>
                    <table class="vitals-table" id="vitalsTable" style="display:none;">
                        <thead>
                            <tr><th>Field (VN)</th><th>Field (EN)</th><th>Value</th><th>Unit</th></tr>
                        </thead>
                        <tbody id="vitalsBody"></tbody>
                    </table>
                </div>
            </div>
            <!-- Right column -->
            <div>
                <div class="panel-dark" id="jsonPanel" style="min-height:400px;">
                    <div class="panel-dark-header">
                        <h3>JSON Output</h3>
                        <button class="btn-sm" id="downloadBtn" style="display:none;">Download</button>
                    </div>
                    <div class="json-output" id="jsonOutput">
                        <div class="placeholder-text">Process an image to see results here</div>
                    </div>
                </div>
            </div>
        </div>
        <button class="btn btn-primary" id="processBtn" disabled>Process Image</button>
        <div class="spinner" id="spinner1"></div>
        <div class="error-box" id="errorBox1"></div>
        <div class="status-bar" id="statusBar" style="display:none;">
            <span>Engine: <strong id="engineName">-</strong></span>
            <span>Time: <strong id="processTime">-</strong></span>
            <span>Fields: <strong id="fieldsFound">-</strong></span>
        </div>
    </div>

    <!-- TAB 2: Raw OCR Test -->
    <div class="tab-content" id="tab-raw-ocr">
        <div class="panel">
            <div class="drop-zone" id="dropZone2">
                <p>Drop image here or click to browse</p>
                <small>JPG, PNG supported</small>
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
            <div id="promptField" style="margin-top:0.6rem;">
                <label style="font-weight:600; color:#555; font-size:0.82rem; display:block; margin-bottom:0.2rem;">Prompt:</label>
                <textarea id="promptInput" rows="2" style="width:100%; padding:0.5rem; border:1px solid #ddd; border-radius:6px; font-size:0.82rem; font-family:inherit; resize:vertical;">What text do you see in this image? List everything you can read.</textarea>
            </div>
            <button class="btn btn-primary" id="rawOcrBtn" disabled>Get Raw Model Output</button>
        </div>
        <div class="spinner" id="spinner2"></div>
        <div class="error-box" id="errorBox2"></div>
        <div class="panel" id="resultSection2" style="display:none; margin-top:1rem;">
            <div style="margin-bottom:0.6rem;">
                <span class="meta-badge" id="modelUsed"></span>
                <span class="meta-badge" id="promptUsed"></span>
            </div>
            <div class="raw-text-output" id="rawTextOutput"></div>
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
let selectedFile1 = null;
let lastResultData = null;

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
    errorBox1.style.display = 'none';
}

processBtn.addEventListener('click', async () => {
    if (!selectedFile1) return;
    processBtn.disabled = true;
    spinner1.style.display = 'block';
    errorBox1.style.display = 'none';
    document.getElementById('statusBar').style.display = 'none';
    const formData = new FormData();
    formData.append('file', selectedFile1);
    const startTime = performance.now();
    try {
        const resp = await fetch('/process', { method: 'POST', body: formData });
        const elapsed = ((performance.now() - startTime) / 1000).toFixed(1);
        const data = await resp.json();
        if (!resp.ok) { showError1(data.detail || 'Processing failed'); return; }
        displayResult(data, elapsed);
    } catch (err) { showError1('Connection error: ' + err.message); }
    finally { spinner1.style.display = 'none'; processBtn.disabled = false; }
});

function showError1(msg) { errorBox1.textContent = msg; errorBox1.style.display = 'block'; }

function displayResult(data, elapsed) {
    lastResultData = data;
    const vitals = data.vitals || {};
    const meta = data.fields_meta || {};
    const bp = vitals.huyet_ap || {};

    const fields = [
        { key: 'mach', value: vitals.mach },
        { key: 'nhiet_do', value: vitals.nhiet_do },
        { key: 'huyet_ap', value: (bp.tam_thu != null || bp.tam_truong != null) ? ((bp.tam_thu||'-') + '/' + (bp.tam_truong||'-')) : null },
        { key: 'nhip_tho', value: vitals.nhip_tho },
        { key: 'can_nang', value: vitals.can_nang },
        { key: 'chieu_cao', value: vitals.chieu_cao },
        { key: 'spo2', value: vitals.spo2 },
    ];

    let foundCount = 0;
    const tbody = document.getElementById('vitalsBody');
    tbody.innerHTML = fields.map(f => {
        const info = meta[f.key] || {};
        const isNull = f.value == null;
        if (!isNull) foundCount++;
        return '<tr>' +
            '<td>' + (info.label_vn || f.key) + '</td>' +
            '<td>' + (info.label_en || '') + '</td>' +
            '<td class="' + (isNull ? 'null' : 'val') + '">' + (isNull ? '-' : f.value) + '</td>' +
            '<td>' + (info.unit || '') + '</td>' +
            '</tr>';
    }).join('');

    document.getElementById('vitalsTable').style.display = 'table';
    document.getElementById('engineName').textContent = data.ocr_engine || '-';
    document.getElementById('processTime').textContent = elapsed + 's';
    document.getElementById('fieldsFound').textContent = foundCount + '/7';
    document.getElementById('statusBar').style.display = 'flex';
    document.getElementById('jsonOutput').innerHTML = syntaxHighlight(JSON.stringify(data, null, 2));
    document.getElementById('downloadBtn').style.display = 'inline-block';
}

document.getElementById('downloadBtn').addEventListener('click', () => {
    if (!lastResultData) return;
    const blob = new Blob([JSON.stringify(lastResultData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = (lastResultData.source_image || 'result').replace(/\.[^.]+$/, '') + '.json';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
});

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
        document.getElementById('modelUsed').textContent = 'Model: ' + data.model;
        document.getElementById('promptUsed').textContent = 'Prompt: ' + (data.prompt_used || '').substring(0, 50) + (data.prompt_used && data.prompt_used.length > 50 ? '...' : '');
        document.getElementById('rawTextOutput').textContent = data.raw_response || '(empty)';
        resultSection2.style.display = 'block';
    } catch (err) { showError2('Connection error: ' + err.message); }
    finally { spinner2.style.display = 'none'; rawOcrBtn.disabled = false; }
});

function showError2(msg) { errorBox2.textContent = msg; errorBox2.style.display = 'block'; }

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
