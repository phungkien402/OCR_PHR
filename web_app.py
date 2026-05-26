"""FastAPI server for OCR Vital Signs — optimized for mobile API calls.

Changes vs original:
- Async Ollama call (httpx) — does not block event loop
- CORS enabled — required for Android/mobile clients
- preprocess_for_vlm() instead of slow preprocess_image() on Qwen path
- /health endpoint for connectivity check from mobile
- Cleaner error messages

Usage:
    pip install httpx
    uvicorn web_app:app --host 0.0.0.0 --port 8502
"""

import os
import tempfile
import time

import cv2
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

app = FastAPI(title="OCR Vital Signs")

# CORS — allow any origin so Android app (and LAN IPs) can call freely
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# Health check
# ─────────────────────────────────────────────

@app.get("/health")
async def health():
    """Lightweight endpoint for mobile connectivity check."""
    return {"status": "ok"}


# ─────────────────────────────────────────────
# Main OCR endpoint
# ─────────────────────────────────────────────

@app.post("/process")
async def process(file: UploadFile = File(...)):
    """Process an uploaded image through the full OCR vital signs pipeline.

    Uses async Ollama call to avoid blocking the event loop.
    """
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        return JSONResponse(
            status_code=400,
            content={"detail": "Only JPG and PNG images are supported."},
        )

    suffix = ".png" if "png" in (file.content_type or "") else ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    t0 = time.perf_counter()
    try:
        from ocr_vitals.main import process_image_async
        result = await process_image_async(tmp_path, device="cuda:0")
        result["processing_time_s"] = round(time.perf_counter() - t0, 2)
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"detail": f"Processing error: {str(e)}"},
        )
    finally:
        os.unlink(tmp_path)


# ─────────────────────────────────────────────
# Raw model output (debug/test)
# ─────────────────────────────────────────────

@app.post("/raw-model-output")
async def raw_model_output(
    file: UploadFile = File(...),
    model: str = Form("qwen3_vl"),
    prompt: str = Form("What text do you see in this image? List everything you can read."),
):
    """Get raw model response with no parsing."""
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        return JSONResponse(status_code=400, content={"detail": "Only JPG/PNG supported."})

    suffix = ".png" if "png" in (file.content_type or "") else ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        if model == "qwen3_vl":
            import base64
            import httpx

            img = cv2.imread(tmp_path)
            if img is None:
                return JSONResponse(status_code=400, content={"detail": "Cannot read image."})

            from ocr_vitals.ocr_engine import _resize_for_vlm, _image_to_b64
            img_b64 = _image_to_b64(_resize_for_vlm(img))

            payload = {
                "model": "qwen3-vl:4b",
                "messages": [{"role": "user", "content": prompt, "images": [img_b64]}],
                "stream": False,
                "options": {"temperature": 0, "num_predict": 512},
            }
            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.post("http://localhost:11434/api/chat", json=payload)
                resp.raise_for_status()
                raw = resp.json()["message"]["content"]

            from ocr_vitals.ocr_engine import _strip_think
            return JSONResponse(content={
                "raw_response": _strip_think(raw),
                "raw_with_think": raw,
                "model": "qwen3-vl:4b (Ollama)",
                "prompt_used": prompt,
            })

        else:  # vietocr
            from PIL import Image
            from ocr_vitals.ocr_engine import load_vietocr
            predictor = load_vietocr("cuda:0")
            raw = predictor.predict(Image.open(tmp_path).convert("RGB"))
            return JSONResponse(content={
                "raw_response": raw,
                "model": "VietOCR vgg_transformer",
                "prompt_used": "(n/a)",
            })

    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": str(e)})
    finally:
        os.unlink(tmp_path)


# ─────────────────────────────────────────────
# Frontend (unchanged)
# ─────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    return _HTML


_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>OCR Vital Signs</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#f5f7fa;color:#333;padding:1.5rem}
.container{max-width:1100px;margin:0 auto}
h1{text-align:center;margin-bottom:.3rem;color:#1a1a2e;font-size:1.6rem}
.subtitle{text-align:center;color:#666;margin-bottom:1.5rem;font-size:.9rem}
.two-col{display:grid;grid-template-columns:1.5fr 1fr;gap:1.2rem}
@media(max-width:768px){.two-col{grid-template-columns:1fr}}
.panel{background:#fff;border-radius:10px;padding:1.5rem;box-shadow:0 2px 10px rgba(0,0,0,.06)}
.dark{background:#1e1e2e;border-radius:10px;padding:1.2rem;display:flex;flex-direction:column;min-height:380px}
.dark h3{color:#cdd6f4;font-size:.9rem;margin-bottom:.8rem}
.drop{border:2px dashed #ccc;border-radius:8px;padding:2.5rem;text-align:center;cursor:pointer;color:#888;transition:.2s}
.drop:hover,.drop.over{border-color:#4a90d9;background:#f0f7ff;color:#4a90d9}
.preview{margin-top:1rem;text-align:center;display:none}
.preview img{max-width:100%;max-height:460px;border-radius:8px;border:1px solid #eee}
.tbl{width:100%;border-collapse:collapse;margin-top:1rem;font-size:.85rem}
.tbl th{text-align:left;padding:.5rem .6rem;border-bottom:2px solid #e0e0e0;color:#555;font-size:.8rem}
.tbl td{padding:.5rem .6rem;border-bottom:1px solid #f0f0f0}
.tbl .v{font-weight:700;color:#1a1a2e}
.tbl .n{color:#ccc}
.btn{display:block;width:100%;padding:.85rem;border:none;border-radius:8px;font-size:1rem;font-weight:700;cursor:pointer;margin-top:1.2rem;background:#4a90d9;color:#fff;transition:.2s}
.btn:hover{background:#357abd}
.btn:disabled{background:#ccc;cursor:not-allowed}
.spin{display:none;text-align:center;padding:1.5rem}
.spin::after{content:'';display:inline-block;width:36px;height:36px;border:4px solid #eee;border-top-color:#4a90d9;border-radius:50%;animation:sp .8s linear infinite}
@keyframes sp{to{transform:rotate(360deg)}}
.err{display:none;background:#fef2f2;border:1px solid #fecaca;border-radius:8px;padding:.8rem 1.2rem;color:#dc2626;margin-top:1rem;font-size:.85rem}
.json-out{flex:1;overflow-y:auto;color:#cdd6f4;font-family:monospace;font-size:.78rem;line-height:1.5;white-space:pre-wrap;word-break:break-word}
.stat{display:flex;gap:1.5rem;align-items:center;justify-content:center;margin-top:1rem;font-size:.82rem;color:#666}
.stat strong{color:#1a1a2e}
.ph{color:#6c7086;font-size:.8rem;text-align:center;padding:3rem 1rem}
input[type=file]{display:none}
</style>
</head>
<body>
<div class="container">
<h1>OCR Vital Signs</h1>
<p class="subtitle">Qwen3-VL:4b · JSON output · async</p>
<div class="two-col">
<div>
<div class="panel">
  <div class="drop" id="dz">Drop image here or click to browse<br><small>JPG / PNG</small></div>
  <input type="file" id="fi" accept="image/jpeg,image/png">
  <div class="preview" id="pv"><img id="pi" alt="preview"><p id="fn" style="font-size:.8rem;color:#666;margin-top:.4rem"></p></div>
  <table class="tbl" id="tbl" style="display:none">
    <thead><tr><th>Chỉ số</th><th>Giá trị</th><th>Đơn vị</th></tr></thead>
    <tbody id="tb"></tbody>
  </table>
  <div class="stat" id="sb" style="display:none;margin-top:.8rem">
    Engine: <strong id="en">-</strong> &nbsp;|&nbsp;
    Time: <strong id="pt">-</strong> &nbsp;|&nbsp;
    Fields: <strong id="ff">-</strong>
  </div>
  <button class="btn" id="pb" disabled style="margin-top:1rem">Process Image</button>
  <div class="spin" id="sp"></div>
  <div class="err" id="er"></div>
</div>
</div>
<div>
<div class="dark" id="jp">
  <h3>JSON Output</h3>
  <div class="json-out" id="jo"><div class="ph">Process an image to see results</div></div>
</div>
</div>
</div>
</div>
<script>
const dz=document.getElementById('dz'),fi=document.getElementById('fi'),pv=document.getElementById('pv'),
  pi=document.getElementById('pi'),fn=document.getElementById('fn'),pb=document.getElementById('pb'),
  sp=document.getElementById('sp'),er=document.getElementById('er'),sb=document.getElementById('sb');
let f=null,last=null;
dz.onclick=()=>fi.click();
dz.ondragover=e=>{e.preventDefault();dz.classList.add('over')};
dz.ondragleave=()=>dz.classList.remove('over');
dz.ondrop=e=>{e.preventDefault();dz.classList.remove('over');if(e.dataTransfer.files[0])load(e.dataTransfer.files[0])};
fi.onchange=()=>{if(fi.files[0])load(fi.files[0])};
function load(file){
  if(!file.type.match(/image\/(jpeg|png)/)){er.textContent='JPG or PNG only';er.style.display='block';return}
  f=file;const r=new FileReader();
  r.onload=e=>{pi.src=e.target.result;pv.style.display='block';fn.textContent=file.name+' ('+(file.size/1024).toFixed(1)+' KB)'};
  r.readAsDataURL(file);pb.disabled=false;er.style.display='none';
}
pb.onclick=async()=>{
  if(!f)return;pb.disabled=true;sp.style.display='block';er.style.display='none';sb.style.display='none';
  const fd=new FormData();fd.append('file',f);
  const t0=performance.now();
  try{
    const r=await fetch('/process',{method:'POST',body:fd});
    const d=await r.json();
    if(!r.ok){er.textContent=d.detail||'Error';er.style.display='block';return}
    show(d,(performance.now()-t0)/1000);
  }catch(e){er.textContent='Connection error: '+e.message;er.style.display='block'}
  finally{sp.style.display='none';pb.disabled=false}
};
const META={mach:{n:'Mạch',u:'BPM'},nhiet_do:{n:'Nhiệt độ',u:'°C'},huyet_ap:{n:'Huyết áp',u:'mmHg'},
  nhip_tho:{n:'Nhịp thở',u:'lần/ph'},can_nang:{n:'Cân nặng',u:'kg'},chieu_cao:{n:'Chiều cao',u:'cm'},spo2:{n:'SpO2',u:'%'}};
function show(d,elapsed){
  last=d;const v=d.vitals||{},bp=v.huyet_ap||{};let found=0;
  const rows=[
    {k:'mach',val:v.mach},{k:'nhiet_do',val:v.nhiet_do},
    {k:'huyet_ap',val:(bp.tam_thu!=null||bp.tam_truong!=null)?(bp.tam_thu||'-')+'/'+(bp.tam_truong||'-'):null},
    {k:'nhip_tho',val:v.nhip_tho},{k:'can_nang',val:v.can_nang},{k:'chieu_cao',val:v.chieu_cao},{k:'spo2',val:v.spo2}
  ];
  document.getElementById('tb').innerHTML=rows.map(r=>{
    const m=META[r.k]||{};const nil=r.val==null;if(!nil)found++;
    return`<tr><td>${m.n||r.k}</td><td class="${nil?'n':'v'}">${nil?'-':r.val}</td><td>${m.u||''}</td></tr>`;
  }).join('');
  document.getElementById('tbl').style.display='table';
  document.getElementById('en').textContent=d.ocr_engine||'-';
  document.getElementById('pt').textContent=(d.processing_time_s||elapsed.toFixed(1))+'s';
  document.getElementById('ff').textContent=found+'/7';
  sb.style.display='flex';
  document.getElementById('jo').innerHTML=hl(JSON.stringify(d,null,2));
}
function hl(s){
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
    .replace(/("(\\u[\da-fA-F]{4}|\\[^u]|[^\\"])*"(\s*:)?|\b(true|false|null)\b|-?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?)/g,
    m=>`<span style="color:${/^"/.test(m)?/:$/.test(m)?'#89b4fa':'#a6e3a1':/true|false/.test(m)?'#cba6f7':'#fab387'}">${m}</span>`);
}
</script>
</body>
</html>"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8502)
