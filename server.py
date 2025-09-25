import os
import requests
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

# ===== Roboflow (Hosted Object Detection) =====
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "").strip()
RF_HOST   = os.getenv("RF_HOST", "https://detect.roboflow.com").rstrip("/")
RF_PROJECT = os.getenv("RF_PROJECT", "fresh-or-rotten-detection-1yxeg").strip()
RF_VERSION = os.getenv("RF_VERSION", "1").strip()

RF_URL = f"{RF_HOST}/{RF_PROJECT}/{RF_VERSION}"  # e.g. https://detect.roboflow.com/fresh-or-rotten-detection-1yxeg/1
if not ROBOFLOW_API_KEY:
    raise RuntimeError("Missing ROBOFLOW_API_KEY env var")

# ===== FastAPI app =====
app = FastAPI(title="Fresh or Rotten (Roboflow) • FastAPI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

@app.get("/healthz")
def healthz():
    return {"ok": True}

# --- helper: draw-friendly output (top class + bbox list) ---
def normalize_detection(j):
    """
    Roboflow Hosted detection returns:
      {"predictions":[{"class":"RottenCucumber","x":..,"y":..,"width":..,"height":..,"confidence":0.76}, ...]}
    This normalizer returns the same list and adds a quick top label/confidence.
    """
    preds = j.get("predictions", []) if isinstance(j, dict) else []
    top = None
    if preds:
        top_pred = max(preds, key=lambda p: p.get("confidence", 0.0))
        top = {"label": top_pred.get("class", "?"),
               "confidence": round(100*float(top_pred.get("confidence", 0.0)), 1)}
    return {"top": top, "predictions": preds}

# --- /predict: send file to Roboflow and return JSON ---
@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    data = await image.read()
    files = {"file": ("image.jpg", data, image.content_type or "image/jpeg")}
    # Send API key both as header and query (some setups prefer one or the other)
    headers = {"Authorization": f"Bearer {ROBOFLOW_API_KEY}", "Accept": "application/json"}
    url = f"{RF_URL}?api_key={ROBOFLOW_API_KEY}"
    try:
        r = requests.post(url, headers=headers, files=files, timeout=60)
        r.raise_for_status()
        j = r.json()
    except requests.exceptions.RequestException as e:
        return JSONResponse({"error": "roboflow_request_failed", "detail": str(e)}, status_code=502)
    except ValueError:
        return JSONResponse({"error": "roboflow_non_json", "detail": r.text[:500]}, status_code=502)
    return JSONResponse(normalize_detection(j))

# --- minimal UI that draws boxes ---
@app.get("/", response_class=HTMLResponse)
def ui():
    return """
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Fresh or Rotten • Demo</title>
  <style>
    body{font-family:system-ui,Segoe UI,Arial; margin:20px; }
    .row{display:flex; gap:16px; align-items:flex-start; flex-wrap:wrap}
    canvas,img{max-width:100%; border:1px solid #ddd; border-radius:8px}
    .pill{display:inline-block; padding:6px 10px; border-radius:999px; background:#eef; font-weight:700}
  </style>
</head>
<body>
  <h1>🍎 Fresh or Rotten — Roboflow Hosted (YOLOv12)</h1>
  <p class="pill">Project: fresh-or-rotten-detection-1yxeg / Version: 1</p>
  <div class="row" style="margin-top:12px">
    <input id="file" type="file" accept="image/*"/>
    <button onclick="predict()">Predict</button>
  </div>
  <div class="row" style="margin-top:16px">
    <img id="preview" style="max-width:420px; display:none"/>
    <canvas id="canvas" width="420" height="420" style="display:none"></canvas>
  </div>
  <pre id="out" style="background:#111;color:#eee;padding:10px;border-radius:8px;max-width:780px;overflow:auto"></pre>

<script>
const img = document.getElementById('preview');
const cnv = document.getElementById('canvas');
const ctx = cnv.getContext('2d');
const out = document.getElementById('out');
const file = document.getElementById('file');

function drawBoxes(preds){
  if(!img.src) return;
  const scale = Math.min(cnv.width / img.naturalWidth, cnv.height / img.naturalHeight);
  const w = img.naturalWidth * scale, h = img.naturalHeight * scale;
  cnv.width = w; cnv.height = h;
  ctx.drawImage(img, 0, 0, w, h);
  ctx.lineWidth = 2; ctx.font = "12px system-ui"; ctx.textBaseline = "top";
  preds.forEach(p=>{
    const x = (p.x - p.width/2) * scale;
    const y = (p.y - p.height/2) * scale;
    const w2 = p.width * scale, h2 = p.height * scale;
    ctx.strokeStyle = "#00c2ff"; ctx.fillStyle = "rgba(0,194,255,0.15)";
    ctx.fillRect(x,y,w2,h2); ctx.strokeRect(x,y,w2,h2);
    ctx.fillStyle = "#003d4d";
    const label = `${p.class} ${(p.confidence*100).toFixed(1)}%`;
    ctx.fillRect(x, y-14, ctx.measureText(label).width+6, 14);
    ctx.fillStyle = "#fff"; ctx.fillText(label, x+3, y-13);
  });
}

async function predict(){
  const f = file.files[0];
  if(!f){ alert('Choose an image'); return; }
  img.src = URL.createObjectURL(f);
  img.style.display = "block"; cnv.style.display = "block";
  await new Promise(res=> img.onload = res);

  const fd = new FormData();
  fd.append('image', f, f.name);
  const r = await fetch('/predict', {method:'POST', body:fd});
  const j = await r.json();
  out.textContent = JSON.stringify(j, null, 2);
  drawBoxes(j.predictions || []);
}
</script>
</body>
</html>
    """
