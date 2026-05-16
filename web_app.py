"""Web interface for OCR Vital Signs pipeline.

Usage:
    uvicorn web_app:app --host 0.0.0.0 --port 8080
"""

import json
import os
import tempfile

from fastapi import FastAPI, File, UploadFile
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
.vitals-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 1rem; margin-bottom: 1.5rem; }
.vital-card { background: #f8fafc; border-radius: 8px; padding: 1rem; text-align: center; border: 1px solid #e8ecf0; }
.vital-card .label { font-size: 0.75rem; text-transform: uppercase; color: #888; margin-bottom: 0.3rem; }
.vital-card .value { font-size: 1.5rem; font-weight: 700; color: #1a1a2e; }
.vital-card .value.null { color: #ccc; font-size: 1rem; }
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
input[type="file"] { display: none; }
</style>
</head>
<body>
<div class="container">
    <h1>OCR Vital Signs</h1>
    <p class="subtitle">Upload a blood pressure monitor image to extract vital signs</p>
    <div class="card">
        <div class="drop-zone" id="dropZone">
            <p>Drop image here or click to browse</p>
            <small>Supports JPG, PNG</small>
        </div>
        <input type="file" id="fileInput" accept="image/jpeg,image/png">
        <div class="preview-container" id="previewContainer">
            <img id="previewImg" alt="Preview">
            <p class="filename" id="fileName"></p>
        </div>
        <button class="btn btn-primary" id="processBtn" disabled>Process</button>
    </div>
    <div class="spinner" id="spinner"></div>
    <div class="error-box" id="errorBox"></div>
    <div class="card result-section" id="resultSection">
        <div class="result-header"><h2>Extracted Vitals</h2></div>
        <div class="vitals-grid" id="vitalsGrid"></div>
        <h2 style="font-size:1rem; margin-bottom:0.8rem; color:#1a1a2e;">Raw JSON</h2>
        <div class="json-output" id="jsonOutput"></div>
    </div>
</div>
<script>
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const previewContainer = document.getElementById('previewContainer');
const previewImg = document.getElementById('previewImg');
const fileName = document.getElementById('fileName');
const processBtn = document.getElementById('processBtn');
const spinner = document.getElementById('spinner');
const errorBox = document.getElementById('errorBox');
const resultSection = document.getElementById('resultSection');
const vitalsGrid = document.getElementById('vitalsGrid');
const jsonOutput = document.getElementById('jsonOutput');
let selectedFile = null;

dropZone.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.classList.add('dragover'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
dropZone.addEventListener('drop', (e) => {
    e.preventDefault(); dropZone.classList.remove('dragover');
    if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener('change', () => { if (fileInput.files.length) handleFile(fileInput.files[0]); });

function handleFile(file) {
    if (!file.type.match(/^image\/(jpeg|png)$/)) { showError('Please upload a JPG or PNG image.'); return; }
    selectedFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImg.src = e.target.result;
        previewContainer.style.display = 'block';
        fileName.textContent = file.name + ' (' + (file.size / 1024).toFixed(1) + ' KB)';
    };
    reader.readAsDataURL(file);
    processBtn.disabled = false;
    resultSection.style.display = 'none';
    errorBox.style.display = 'none';
}

processBtn.addEventListener('click', async () => {
    if (!selectedFile) return;
    processBtn.disabled = true;
    spinner.style.display = 'block';
    resultSection.style.display = 'none';
    errorBox.style.display = 'none';
    const formData = new FormData();
    formData.append('file', selectedFile);
    try {
        const resp = await fetch('/process', { method: 'POST', body: formData });
        const data = await resp.json();
        if (!resp.ok) { showError(data.detail || 'Processing failed'); return; }
        displayResult(data);
    } catch (err) { showError('Connection error: ' + err.message); }
    finally { spinner.style.display = 'none'; processBtn.disabled = false; }
});

function showError(msg) { errorBox.textContent = msg; errorBox.style.display = 'block'; }

function displayResult(data) {
    const vitals = data.vitals || {};
    const bp = vitals.huyet_ap || {};
    const cards = [
        { label: 'SYS (mmHg)', value: bp.tam_thu },
        { label: 'DIA (mmHg)', value: bp.tam_truong },
        { label: 'Pulse (/min)', value: vitals.mach },
        { label: 'Temp (°C)', value: vitals.nhiet_do },
        { label: 'SpO2 (%)', value: vitals.spo2 },
        { label: 'Resp Rate', value: vitals.nhip_tho },
    ];
    vitalsGrid.innerHTML = cards.map(c => {
        const isNull = c.value == null;
        return '<div class="vital-card"><div class="label">' + c.label +
            '</div><div class="value ' + (isNull ? 'null' : '') + '">' +
            (isNull ? '—' : c.value) + '</div></div>';
    }).join('');
    jsonOutput.innerHTML = syntaxHighlight(JSON.stringify(data, null, 2));
    resultSection.style.display = 'block';
}

function syntaxHighlight(json) {
    return json.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
        .replace(/("(\u[a-zA-Z0-9]{4}|\[^u]|[^\"])*"(\s*:)?|(true|false)|-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?|null)/g,
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
    """Process an uploaded image through the OCR vital signs pipeline."""
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
