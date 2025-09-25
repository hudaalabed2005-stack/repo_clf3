import os
from datetime import datetime, timedelta
import sqlite3
from pathlib import Path
import io, csv
import requests
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

# ------------------ Roboflow (Object Detection) ------------------
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "")
PROJECT = os.getenv("RF_PROJECT", "fresh-or-rotten-detection-1yxeg").strip()
VERSION = os.getenv("RF_VERSION", "1").strip()
# Detect (object detection) endpoint:
DETECT_URL = f"https://detect.roboflow.com/{PROJECT}/{VERSION}"

if not ROBOFLOW_API_KEY:
    raise RuntimeError("Missing ROBOFLOW_API_KEY environment variable.")

# ------------------ FastAPI + CORS ------------------
app = FastAPI(title="Fruit & Gas Cloud API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------ In-memory cache ------------------
LAST = {
    "vision": None,
    "vision_updated": None,
    "gas": None,
    "gas_updated": None,
}

# ------------------ SQLite for gas history ------------------
DB_PATH = Path("data.db")

def init_db():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS gas_readings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            co2 REAL, nh3 REAL, benzene REAL, alcohol REAL
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_gas_ts ON gas_readings(ts)")
    con.commit()
    con.close()

def save_reading(ppm: dict):
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute(
        "INSERT INTO gas_readings (ts, co2, nh3, benzene, alcohol) VALUES (?, ?, ?, ?, ?)",
        (datetime.utcnow().isoformat(),
         ppm.get("co2"), ppm.get("nh3"),
         ppm.get("benzene"), ppm.get("alcohol"))
    )
    con.commit()
    con.close()

def load_history_last_days(days: int = 2):
    cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("""
        SELECT ts, co2, nh3, benzene, alcohol
        FROM gas_readings
        WHERE ts >= ?
        ORDER BY ts ASC
    """, (cutoff,))
    rows = cur.fetchall()
    con.close()
    return [
        {"time": ts, "ppm": {"co2": co2, "nh3": nh3, "benzene": benz, "alcohol": alc}}
        for (ts, co2, nh3, benz, alc) in rows
    ]

init_db()

# ------------------ Vision helpers (Object Detection) ------------------
def extract_top_detection(resp_obj):
    """
    From Roboflow Detect response -> pick highest-confidence detection.
    Expected shape:
      {"predictions":[{"class":"RottenCucumber","confidence":0.76, ...}, ...], ...}
    Returns: {"label": str, "confidence": float} or None
    """
    if not resp_obj or "predictions" not in resp_obj:
        return None
    preds = resp_obj.get("predictions") or []
    if not isinstance(preds, list) or not preds:
        return None
    top = max(preds, key=lambda p: float(p.get("confidence", 0.0)))
    label = str(top.get("class", "?"))
    conf = float(top.get("confidence", 0.0)) * 100.0
    return {"label": label, "confidence": round(conf, 1)}

# ------------------ /predict (Detect) ------------------
@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    """
    Accept an uploaded image, call Roboflow Detect (object detection),
    cache the raw response, return JSON or error.
    """
    data = await image.read()
    try:
        headers = {
            "Authorization": f"Bearer {ROBOFLOW_API_KEY}",
            "Accept": "application/json"
        }
        files = {"file": ("image.jpg", data, image.content_type or "image/jpeg")}
        # You can add params like confidence/overlap if you want:
        # params = {"confidence": 0.5, "overlap": 0.3}
        resp = requests.post(DETECT_URL, headers=headers, files=files, timeout=60)
        if resp.status_code == 403:
            return JSONResponse(
                {
                    "error": "roboflow_403",
                    "detail": "Roboflow detect forbidden (403). Check API key, project path, and Hosted API deploy.",
                    "endpoint": DETECT_URL
                },
                status_code=502
            )
        resp.raise_for_status()
        j = resp.json()
    except requests.exceptions.RequestException as e:
        return JSONResponse(
            {"error": "roboflow_request_failed", "detail": str(e), "endpoint": DETECT_URL},
            status_code=502
        )
    except ValueError:
        return JSONResponse(
            {"error": "roboflow_non_json", "detail": resp.text[:500], "endpoint": DETECT_URL},
            status_code=502
        )

    LAST["vision"] = j
    LAST["vision_updated"] = datetime.utcnow().isoformat()
    return JSONResponse(j)

# ------------------ Gas model ------------------
class GasReading(BaseModel):
    vrl: float | None = None
    adc: int | None = None
    adc_max: int | None = 4095
    vref: float | None = 3.3
    rl:   float | None = 10000.0
    rs:   float | None = None
    r0:   float | None = None

def _ppm_from_ratio(ratio: float, a: float, b: float) -> float:
    if ratio is None or ratio <= 0:
        return 0.0
    return max(0.0, a * (ratio ** b))

@app.post("/gas")
def gas(g: GasReading):
    """
    Accept gas info from ESP32/UNO (or the manual UI),
    compute Rs/ratio/ppm, update LAST, and persist to DB.
    """
    VREF = float(g.vref or 3.3)
    RL   = float(g.rl or 10000.0)

    if g.vrl is None and g.adc is not None:
        adc_max = int(g.adc_max or 4095)
        g.vrl = (float(g.adc) / float(adc_max)) * VREF

    if g.vrl is None and g.rs is None:
        return {"error": "Send at least one of: vrl, adc, or rs."}

    rs = float(g.rs) if g.rs is not None else ((VREF - float(g.vrl)) * RL) / max(0.001, float(g.vrl))
    r0 = float(g.r0) if g.r0 is not None else rs
    ratio = rs / max(1e-6, r0)

    data = {
        "vrl": round(float(g.vrl), 3) if g.vrl is not None else None,
        "rs": round(rs, 1),
        "r0": round(r0, 1),
        "ratio": round(ratio, 3),
        "ppm": {
            "co2":     round(_ppm_from_ratio(ratio, 116.6021, -2.7690), 1),
            "nh3":     round(_ppm_from_ratio(ratio, 102.6940, -2.4880), 1),
            "benzene": round(_ppm_from_ratio(ratio, 76.63,   -2.1680), 1),
            "alcohol": round(_ppm_from_ratio(ratio, 77.255,  -3.18),   1),
        },
    }

    LAST["gas"] = data
    LAST["gas_updated"] = datetime.utcnow().isoformat()
    save_reading(data["ppm"])
    return {"ok": True, "data": data}

@app.post("/cron/snapshot")
def cron_snapshot():
    if not LAST.get("gas") or not LAST["gas"].get("ppm"):
        return {"ok": False, "error": "No gas reading to snapshot yet."}
    save_reading(LAST["gas"]["ppm"])
    return {"ok": True, "saved": LAST["gas"]["ppm"]}

@app.get("/history")
def history():
    return {"history": load_history_last_days(days=2)}

# ------------------ Summary (vision + gas) ------------------
def _summarize(last: dict) -> dict:
    pred = extract_top_detection(last.get("vision"))

    gas = (last.get("gas") or {}).get("ppm", {})
    co2  = gas.get("co2")
    nh3  = gas.get("nh3")
    benz = gas.get("benzene")
    alco = gas.get("alcohol")

    # Thresholds – tune as needed
    co2_hi = (co2 is not None) and (co2 >= 2000)
    nh3_hi = (nh3 is not None) and (nh3 >= 15)
    voc_hi = (benz or 0) >= 5 or (alco or 0) >= 10

    # Treat any detection whose class contains 'rotten' as spoiled
    model_rotten = bool(pred and isinstance(pred.get("label"), str) and ("rotten" in pred["label"].lower()))
    spoiled = model_rotten or co2_hi or nh3_hi or voc_hi

    return {
        "vision": pred,  # {"label":"RottenCucumber","confidence":76.0}
        "gas_ppm": {"co2": co2, "nh3": nh3, "benzene": benz, "alcohol": alco},
        "gas_flags": {"co2_high": co2_hi, "nh3_high": nh3_hi, "voc_high": voc_hi},
        "decision": "SPOILED" if spoiled else "FRESH",
    }

@app.get("/export.csv")
def export_csv():
    rows = load_history_last_days(days=2)
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["timestamp_utc", "co2_ppm", "nh3_ppm", "benzene_ppm", "alcohol_eq"])
    for r in rows:
        ts = r["time"]
        ppm = r["ppm"] or {}
        w.writerow([ts, ppm.get("co2"), ppm.get("nh3"), ppm.get("benzene"), ppm.get("alcohol")])
    csv_data = buf.getvalue()
    return HTMLResponse(
        content=csv_data,
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="gas_last_2_days.csv"'}
    )

@app.get("/summary")
def summary():
    return _summarize(LAST)

# ------------------ UI ------------------
@app.get("/", response_class=HTMLResponse)
def welcome():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Welcome • Fruit Detector</title>
  <style>
    body{
      margin:0; font-family:Inter,Arial,Helvetica,sans-serif; color:#fff; min-height:100vh;
      background:url('https://i.pinimg.com/originals/30/ab/43/30ab43926be6852d3b03572459ab847d.gif') center/cover no-repeat fixed;
      display:flex; align-items:center; justify-content:center;
    }
    body::before{content:""; position:fixed; inset:0; background:linear-gradient(to bottom right,rgba(0,0,0,.35),rgba(0,0,0,.55)); pointer-events:none;}
    .wrap{
      position:relative; text-align:center; padding:48px 40px; max-width:900px; width:92%;
      background:rgba(0,0,0,.30); border:1px solid rgba(255,255,255,.15); border-radius:20px; box-shadow:0 20px 60px rgba(0,0,0,.45);
      backdrop-filter:blur(6px); -webkit-backdrop-filter:blur(6px);
    }
    h1{font-size:clamp(2rem,4vw,3rem); margin:0 0 12px; letter-spacing:.5px;}
    p{font-size:1.05rem; opacity:.95; margin:0 auto 22px; max-width:760px; line-height:1.6}
    .cta{
      display:inline-block; margin-top:8px; padding:14px 26px; font-weight:800; letter-spacing:.2px; color:#0b3d2e;
      background:linear-gradient(135deg,#7CFFCB,#4ADE80); border:none; border-radius:12px; text-decoration:none;
      box-shadow:0 8px 24px rgba(16,185,129,.35); transition:transform .15s, box-shadow .15s, opacity .15s;
    }
    .cta:hover{transform:translateY(-2px); box-shadow:0 12px 28px rgba(16,185,129,.45);}
    .badge{
      display:inline-flex; gap:8px; padding:8px 12px; border-radius:999px; font-weight:700; color:#0b3d2e; background:rgba(255,255,255,.9);
      box-shadow:inset 0 0 0 1px rgba(0,0,0,.06); margin-bottom:14px
    }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="badge">🍎 <span>Fruit Freshness & Gas Detector</span></div>
    <h1>Smarter Food, Fresher Choices</h1>
    <p>Check fruit freshness with AI and estimate air quality using your sensor readings.</p>
    <a class="cta" href="/app">Start Detecting Freshness</a>
    <div class="meta">Or explore the API docs at <code>/docs</code>.</div>
  </div>
</body>
</html>
    """

@app.get("/app", response_class=HTMLResponse)
def ui():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>Fruit Freshness Detector</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.4/dist/chart.umd.min.js"></script>
  <style>
    :root{ --green:#22c55e; --green-dark:#16a34a; --blue:#0ea5e9; --amber:#f59e0b;
           --red:#ef4444; --slate:#1f2937; --glass-bg: rgba(255,255,255,.86);
           --glass-border: rgba(0,0,0,.06); --shadow: 0 10px 30px rgba(0,0,0,.18); }
    body{ margin:0; font-family:Inter,system-ui,Arial,Helvetica,sans-serif; color:var(--slate);
          background:url('https://i.pinimg.com/originals/3d/91/51/3d9151870044e69f2d93a9d0311275dd.gif') center/cover no-repeat fixed; min-height:100vh }
    body::before{content:""; position:fixed; inset:0; background:linear-gradient(180deg,rgba(0,0,0,.32),rgba(0,0,0,.22)); pointer-events:none; z-index:-1}
    header{ background:linear-gradient(90deg, rgba(34,197,94,.97), rgba(22,163,74,.97));
            padding:18px 16px; text-align:center; color:#fff; box-shadow:var(--shadow); position:sticky; top:0; z-index:10; backdrop-filter: blur(4px); }
    header h1{ margin:0; font-size:clamp(1.3rem,2.8vw,1.9rem); letter-spacing:.3px }
    header p{ margin:6px 0 0 0; opacity:.95 }
    .container{ width:92%; max-width:1100px; margin:24px auto; display:grid; gap:18px }
    .card{ background:var(--glass-bg); border-radius:16px; padding:18px; box-shadow:var(--shadow);
           border:1px solid var(--glass-border); backdrop-filter: blur(6px); transition: transform .15s ease, box-shadow .2s ease; }
    .card:hover{ transform:translateY(-1px); box-shadow: 0 12px 34px rgba(0,0,0,.22); }
    .card h2{ margin:0 0 12px; color:var(--green-dark); font-size:1.15rem; border-left:4px solid var(--green-dark); padding-left:10px }
    button{ background:var(--green); color:#fff; padding:10px 14px; border:none; border-radius:10px;
            cursor:pointer; font-weight:800; letter-spacing:.2px; box-shadow: 0 5px 18px rgba(34,197,94,.35);
            display:inline-flex; align-items:center; gap:8px; transition: background .15s ease, transform .1s ease; }
    button:hover{ background:var(--green-dark) } button:active{ transform:translateY(1px) }
    button.secondary{ background:#e7fff1; color:#0b3d2e; border:1px solid #b9f3d2; box-shadow:none; }
    button.gray{ background:#f3f4f6; color:#111827; border:1px solid #e5e7eb; box-shadow:none; }
    input[type=file], input[type=number]{ padding:10px 12px; border:1px solid #d1d5db; border-radius:10px; background:#fff; font-weight:600;
                                          outline:none; transition:border-color .15s ease, box-shadow .15s ease; }
    input[type=file]:focus, input[type=number]:focus{ border-color: var(--green); box-shadow: 0 0 0 3px rgba(34,197,94,.18); }
    .row{ display:flex; gap:10px; flex-wrap:wrap; align-items:center }
    .pill{ padding:6px 10px; border-radius:999px; font-weight:700; font-size:.9rem; border:1px solid #d1d5db; background:#fff }
    .ok{ background:#ecfdf5; color:#065f46; border:1px solid #a7f3d0 } .bad{ background:#fef2f2; color:#991b1b; border:1px solid #fecaca }
    .warn{ background:#fffbeb; color:#92400e; border:1px solid #fde68a } .big{ font-size:22px; font-weight:900; margin-top:10px }
    img,video,canvas{ max-width:100%; border-radius:12px; margin-top:10px } #preview{ display:none; }
    pre{ white-space:pre-wrap; background:#0b1220; color:#e5e7eb; border-radius:12px; padding:12px; max-height:320px; overflow:auto;
         border:1px solid rgba(255,255,255,.05); }
    footer{ text-align:center; padding:16px; background:rgba(238,238,238,.92); color:#111827; margin-top:10px; font-size:.9rem; border-top:1px solid rgba(0,0,0,.06) }
    .chart-wrap{ position:relative; height:280px; width:100%; overflow:hidden; border-radius:12px; background:#ffffffe6; border:1px solid rgba(0,0,0,.06) }
    .chart-empty{ position:absolute; inset:0; display:flex; align-items:center; justify-content:center; color:#6b7280; font-size:.95rem; pointer-events:none; font-weight:700; }
    .toast{ position:fixed; right:14px; bottom:14px; background:rgba(17,24,39,.92); color:#fff; padding:10px 14px; border-radius:10px; font-weight:700;
            box-shadow: var(--shadow); transform: translateY(10px); opacity:0; transition: all .2s ease; z-index:30; }
    .toast.show{ transform: translateY(0); opacity:1; } .status{ font-size:.9rem; opacity:.85; display:inline-flex; align-items:center; gap:8px; margin-left:8px; }
    .dot{ width:8px; height:8px; border-radius:50%; background:#22c55e } .muted{ opacity:.7 }
  </style>
</head>
<body>
  <header>
    <h1>🍎 Fruit Freshness & Gas Detector</h1>
    <p>Upload, predict, and view gas-based decision</p>
  </header>

  <div class="container">

    <div class="card">
      <h2>1) Upload or Capture Fruit Image <span id="visionStatus" class="status muted"><span class="dot"></span> idle</span></h2>
      <div class="row">
        <input id="file" type="file" accept="image/*" />
        <button type="button" onclick="predictFile()">🔮 Predict</button>
        <button type="button" class="secondary" onclick="startCam()">📷 Use Webcam</button>
        <button type="button" class="gray" onclick="snap()">📸 Snapshot</button>
        <button type="button" class="gray" onclick="stopCam()">⏹ Stop Camera</button>
        <button type="button" class="gray" onclick="clearVision()">🧹 Clear Image</button>
      </div>
      <video id="video" autoplay playsinline width="320" height="240" style="display:none;background:#000"></video>
      <canvas id="canvas" width="320" height="240" style="display:none"></canvas>
      <img id="preview" alt="preview" />
      <div id="visionTop" class="big"></div>
      <span id="visionBadge" class="pill" style="display:none"></span>
    </div>

    <div class="card">
      <h2>2) Gas Sensor Reading <span id="gasStatus" class="status muted"><span class="dot" style="background:#0ea5e9"></span> idle</span></h2>
      <div class="row" style="margin-bottom:8px">
        ADC <input id="adc" type="number" value="1800" autocomplete="off" />
        Vref <input id="vref" type="number" value="3.3" step="0.1" autocomplete="off" />
        RL(Ω) <input id="rl" type="number" value="10000" autocomplete="off" />
        R0(Ω) <input id="r0" type="number" value="10000" autocomplete="off" />
        <button type="button" onclick="sendGas()">📤 Send</button>
        <button type="button" class="gray" onclick="preset('fresh')">🍏 Fresh Preset</button>
        <button type="button" class="gray" onclick="preset('spoiled')">🍌 Spoiled Preset</button>
        <button type="button" class="gray" onclick="resetGas()">🔁 Reset</button>
        <button type="button" class="gray" onclick="saveSnap()">💾 Save Snapshot</button>
      </div>
      <div id="gasBadges" style="margin-top:6px"></div>
    </div>

    <div class="card">
      <h2>3) Final Decision</h2>
      <div id="decision" class="big"></div>
      <div class="row" style="margin:10px 0">
        <button type="button" class="secondary" onclick="refresh()">🔄 Refresh Summary</button>
        <button type="button" class="gray" onclick="clearAll()">🧹 Clear All</button>
      </div>
      <pre id="raw"></pre>
    </div>

    <div class="card">
      <h2>4) Gas Chart (Last 2 Days)</h2>
      <div class="chart-wrap">
        <canvas id="gasChart"></canvas>
        <div id="chartEmpty" class="chart-empty">No readings yet — send gas data or save a snapshot.</div>
      </div>
    </div>

  </div>

  <footer>© 2025 Fruit Detector • FastAPI + Roboflow Detect + MQ-135</footer>
  <div id="toast" class="toast">Saved ✔</div>

<script>
"use strict";
const $ = (id) => document.getElementById(id);
const el = {
  file:$('file'), preview:$('preview'), video:$('video'), canvas:$('canvas'),
  visionBadge:$('visionBadge'), visionTop:$('visionTop'),
  gasBadges:$('gasBadges'), decision:$('decision'), raw:$('raw'),
  visionStatus:$('visionStatus'), gasStatus:$('gasStatus'),
  toast:$('toast'), chartEmpty:$('chartEmpty'), gasChart:$('gasChart'),
  adc:$('adc'), vref:$('vref'), rl:$('rl'), r0:$('r0'),
};
const badge = (t, c) => `<span class="pill ${c}">${t}</span>`;
const GAS_LS_KEY = "gas_history_cache_v1";

function toast(msg){ el.toast.textContent = msg || 'Done'; el.toast.classList.add('show'); setTimeout(()=> el.toast.classList.remove('show'), 1500); }
function clearVision(){ el.preview.src=''; el.preview.style.display='none'; el.video.style.display='none'; el.canvas.style.display='none'; el.visionBadge.style.display='none'; el.visionTop.textContent=''; }
function clearAll(){ clearVision(); el.gasBadges.innerHTML=''; el.decision.className='big'; el.decision.textContent=''; el.raw.textContent=''; }
function setStatus(which, state){
  const statusEl = (which === 'vision') ? el.visionStatus : el.gasStatus;
  if(!statusEl) return;
  const color = (which === 'vision') ? '#22c55e' : '#0ea5e9';
  if(state === 'busy'){ statusEl.classList.remove('muted'); statusEl.innerHTML = `<span class="dot" style="background:${color}"></span> working…`; }
  else { statusEl.classList.add('muted'); statusEl.innerHTML = `<span class="dot" style="background:${color}"></span> idle`; }
}

function updateVisionFromRaw(j){
  if (Array.isArray(j?.predictions) && j.predictions.length){
    const pred = [...j.predictions].sort((a,b)=>(b.confidence||0)-(a.confidence||0))[0];
    const lbl = String(pred.class || '?');
    const conf = Number((pred.confidence||0)*100).toFixed(1);
    el.visionBadge.style.display = 'inline-block';
    const bad = /(^|_|\b)rotten/i.test(lbl);
    el.visionBadge.className = 'pill ' + (bad ? 'bad' : 'ok');
    el.visionBadge.textContent = `${lbl} • ${conf}%`;
    el.visionTop.textContent = lbl.replace(/_/g,' ').toUpperCase();
  }
}

async function predictFile(){
  const f = el.file.files[0];
  if (!f) { alert('Choose an image'); return; }
  el.preview.src = URL.createObjectURL(f); el.preview.style.display = 'block';
  const fd = new FormData(); fd.append('image', f, f.name);
  setStatus('vision','busy');
  try {
    const r = await fetch('/predict', { method:'POST', body:fd });
    let j = null; try { j = await r.json(); } catch (_) { j = {error:'Bad JSON from server'}; }
    if (!r.ok || j?.error){ alert('Predict failed: ' + (j?.error || r.statusText)); return; }
    updateVisionFromRaw(j); await refresh();
  } finally { setStatus('vision','idle'); }
}

let stream = null;
async function startCam(){ try{ stream = await navigator.mediaDevices.getUserMedia({video:true}); el.video.srcObject = stream; el.video.style.display='block'; }catch(e){ alert('Camera error: '+e); } }
function stopCam(){ if(stream){ stream.getTracks().forEach(t=>t.stop()); stream=null; } el.video.style.display='none'; }
function snap(){
  if (!stream) { alert('Start the webcam first'); return; }
  const ctx = el.canvas.getContext('2d'); el.canvas.style.display='block';
  ctx.drawImage(el.video, 0, 0, el.canvas.width, el.canvas.height);
  el.canvas.toBlob(async (b) => {
    const fd = new FormData(); fd.append('image', b, 'snapshot.jpg');
    setStatus('vision','busy');
    try {
      const r = await fetch('/predict', { method:'POST', body:fd });
      let j = null; try { j = await r.json(); } catch (_) { j = {error:'Bad JSON from server'}; }
      if (!r.ok || j?.error){ alert('Predict failed: ' + (j?.error || r.statusText)); return; }
      updateVisionFromRaw(j); await refresh();
    } finally { setStatus('vision','idle'); }
  }, 'image/jpeg', 0.92);
}

/* Gas */
async function sendGas(){
  const body = {
    adc: parseInt(el.adc.value || '0'),
    vref: parseFloat(el.vref.value || '3.3'),
    rl: parseInt(el.rl.value || '10000'),
    r0: parseInt(el.r0.value || '10000'),
    adc_max: 4095
  };
  setStatus('gas','busy');
  try{
    const r = await fetch('/gas', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body) });
    if(!r.ok){ const t = await r.text(); alert('Gas send failed: ' + (t || r.statusText)); return; }
    await refresh(); await loadChart(true);
  }finally{ setStatus('gas','idle'); }
}
function resetGas(){ el.adc.value="1800"; el.vref.value="3.3"; el.rl.value="10000"; el.r0.value="10000"; }
function preset(type){ if(type==='fresh'){ el.adc.value="1200"; el.r0.value="12000"; } if(type==='spoiled'){ el.adc.value="2500"; el.r0.value="8000"; } }
async function saveSnap(){ const r = await fetch('/cron/snapshot', {method:'POST'}); const j = await r.json(); if(j.ok){ await loadChart(true); toast('Snapshot saved ✔'); } else { alert('No reading to save yet.'); } }

/* Summary */
async function refresh(){
  const r = await fetch('/summary', {cache:'no-store'}); const s = await r.json();
  if (s.vision && s.vision.label){
    el.visionBadge.style.display='inline-block';
    const lbl = String(s.vision.label);
    const conf = Number(s.vision.confidence ?? 0).toFixed(1);
    const bad = /(^|_|\b)rotten/i.test(lbl);
    el.visionBadge.className = 'pill ' + (bad ? 'bad' : 'ok');
    el.visionBadge.textContent = `${lbl} • ${conf}%`;
    el.visionTop.textContent = lbl.replace(/_/g,' ').toUpperCase();
  } else { el.visionBadge.style.display='none'; el.visionTop.textContent=''; }

  const g  = s.gas_ppm || {}, gf = s.gas_flags || {};
  el.gasBadges.innerHTML = [
    badge(`CO₂ ${g.co2??'—'} ppm`, gf.co2_high ? 'bad' : 'ok'),
    badge(`NH₃ ${g.nh3??'—'} ppm`, gf.nh3_high ? 'bad' : 'ok'),
    badge(`VOC ${g.alcohol??'—'} eq`, gf.voc_high ? 'warn' : 'ok')
  ].join(' ');

  el.decision.className = 'big ' + (s.decision === 'SPOILED' ? 'bad' : 'ok');
  el.decision.textContent = s.decision || '';
  el.raw.textContent = JSON.stringify(s, null, 2);
}
refresh(); setInterval(refresh, 2000);

/* Chart */
let gasChart = null;
function saveCache(rows){ try{ localStorage.setItem("gas_history_cache_v1", JSON.stringify(rows.slice(-300))); }catch(_){} }
function loadCache(){ try{ return JSON.parse(localStorage.getItem("gas_history_cache_v1") || "[]"); }catch(_){ return []; } }
function buildGradient(ctx, color){ const g = ctx.createLinearGradient(0,0,0,ctx.canvas.height); g.addColorStop(0,color+"AA"); g.addColorStop(1,color+"00"); return g; }
async function loadChart(forceFetch=false){
  const canvas = el.gasChart; if(!canvas) return;
  const ctx = canvas.getContext('2d'); if(!ctx) return;
  let rows = [];
  if (!forceFetch){ rows = loadCache(); setTimeout(()=>loadChart(true), 100); }
  else{
    try{ const r = await fetch('/history', {cache:"no-store"}); const j = await r.json(); if(Array.isArray(j.history)) rows = j.history; }catch(_){}
  }
  if (!rows.length) rows = loadCache();
  const labels = rows.map(h => new Date(h.time || h.ts).toLocaleString());
  const co2 = rows.map(h => h?.ppm?.co2 ?? null);
  const nh3 = rows.map(h => h?.ppm?.nh3 ?? null);
  const benz = rows.map(h => h?.ppm?.benzene ?? null);
  el.chartEmpty.style.display = rows.length ? "none" : "flex";
  if(rows.length) saveCache(rows);
  const COL = { co2:"#22c55e", nh3:"#0ea5e9", benz:"#f59e0b" };
  const ds = [
    { label:"CO₂ (ppm)", data:co2, tension:.35, borderColor:COL.co2, pointRadius:0, hitRadius:12, fill:true, backgroundColor:buildGradient(ctx, COL.co2) },
    { label:"NH₃ (ppm)", data:nh3, tension:.35, borderColor:COL.nh3, pointRadius:0, hitRadius:12, fill:true, backgroundColor:buildGradient(ctx, COL.nh3) },
    { label:"Benzene (ppm)", data:benz, tension:.35, borderColor:COL.benz, pointRadius:0, hitRadius:12, fill:true, backgroundColor:buildGradient(ctx, COL.benz) }
  ];
  const options = {
    responsive:true, maintainAspectRatio:false,
    interaction:{ mode:"index", intersect:false },
    plugins:{ legend:{ position:"bottom", labels:{ boxWidth:12, font:{weight:700} } },
              tooltip:{ backgroundColor:"rgba(0,0,0,.85)", titleFont:{weight:800} } },
    scales:{ x:{ ticks:{ autoSkip:true, maxTicksLimit:8 }, grid:{ display:false } },
             y:{ beginAtZero:true, grid:{ color:"rgba(0,0,0,.06)" } } },
    animation:{ duration: 350 }
  };
  if (!gasChart){
    canvas.style.height = "280px";
    gasChart = new Chart(ctx, { type:"line", data:{ labels, datasets: ds }, options });
  }else{
    gasChart.data.labels = labels;
    gasChart.data.datasets[0].data = co2;
    gasChart.data.datasets[1].data = nh3;
    gasChart.data.datasets[2].data = benz;
    gasChart.update();
  }
}
loadChart(false); setInterval(()=>loadChart(true), 60_000);
</script>

</body>
</html>
    """
