from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from joblib import load
from pydantic import BaseModel
import numpy as np, cv2, base64
import json, os

MODEL = None
FEATURE_META = None

app = FastAPI(title="Fruit Ripeness API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# Konfigurasi HSV awal (OpenCV Hue: 0..180) — nanti bisa dikalibrasi
CFG = {
    "banana": {
        "ripe":    ((20, 70, 70), (35,255,255)),
        "unripe":  ((35, 60, 60), (85,255,255)),
        "overripe":(( 0,  0,  0), (30,120,140)),
        "thresholds": {"lo":0.30, "hi":0.70}
    },
    "orange": {
        "ripe":    (( 5,100,70), (20,255,255)),
        "unripe":  ((35, 60,60), (85,255,255)),
        "overripe":(( 0,  0, 0), (25,120,120)),
        "thresholds": {"lo":0.35, "hi":0.65}
    },
    "apple": {
        "ripe_low":  (( 0,120,60), (10,255,255)),
        "ripe_high": ((170,120,60), (180,255,255)),
        "unripe":    ((35, 60,60), (85,255,255)),
        "overripe":  (( 0,  0, 0), (25,120,120)),
        "thresholds": {"lo":0.30, "hi":0.60}
    }
}

def read_image_to_bgr(upload: UploadFile):
    data = np.frombuffer(upload.file.read(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Gambar tidak valid.")
    h,w = img.shape[:2]; scale = 640.0/max(1,w)
    return cv2.resize(img, (640, int(h*scale)))

def segment_fruit(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    # Segmentasi kasar area “berwarna”
    mask = cv2.inRange(hsv, (0,40,40), (180,255,255))
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, 1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, 2)
    cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    fruit_mask = np.zeros_like(mask)
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        cv2.drawContours(fruit_mask, [c], -1, 255, -1)
    return hsv, fruit_mask

def prop(hsv, fruit_mask, lo, hi):
    m = cv2.inRange(hsv, lo, hi)
    m = cv2.bitwise_and(m, fruit_mask)
    return (m>0).sum()

def classify(hsv, fruit_mask, fruit: str):
    total = (fruit_mask>0).sum()
    if total == 0:
        return "unknown", dict(score=0, p_ripe=0, p_unripe=0, p_over=0)

    if fruit=="apple":
        cfg = CFG["apple"]
        p_ripe = (prop(hsv, fruit_mask, *cfg["ripe_low"]) +
                  prop(hsv, fruit_mask, *cfg["ripe_high"])) / total
        p_unripe = prop(hsv, fruit_mask, *cfg["unripe"]) / total
        p_over   = prop(hsv, fruit_mask, *cfg["overripe"]) / total
        lo, hi = cfg["thresholds"]["lo"], cfg["thresholds"]["hi"]
    else:
        cfg = CFG[fruit]
        p_ripe   = prop(hsv, fruit_mask, *cfg["ripe"]) / total
        p_unripe = prop(hsv, fruit_mask, *cfg["unripe"]) / total
        p_over   = prop(hsv, fruit_mask, *cfg["overripe"]) / total
        lo, hi = cfg["thresholds"]["lo"], cfg["thresholds"]["hi"]

    score = p_ripe / max(1e-6, (p_ripe+p_unripe))
    if p_over>0.25: label="overripe"
    elif score<lo:  label="unripe"
    elif score<hi:  label="midripe"
    else:           label="ripe"
    return label, dict(score=float(score), p_ripe=float(p_ripe),
                       p_unripe=float(p_unripe), p_over=float(p_over))

def make_overlay(bgr, fruit_mask, label):
    overlay = bgr.copy()
    color = (0,255,0)
    if label=="ripe": color=(0,180,255)
    if label=="overripe": color=(0,0,255)
    cnts,_ = cv2.findContours(fruit_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, cnts, -1, color, 3)
    cv2.rectangle(overlay, (10,10), (260,60), (0,0,0), -1)
    cv2.putText(overlay, f"Label: {label}", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(255,255,255),2)
    ok, buf = cv2.imencode(".png", overlay)
    if not ok: return None
    return "data:image/png;base64," + base64.b64encode(buf).decode("utf-8")

def try_load_model():
    """Muat model ML (jika tersedia) saat startup."""
    global MODEL, FEATURE_META
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    pkl = os.path.join(models_dir, "model_rf.pkl")
    meta = os.path.join(models_dir, "feature_meta.json")
    if os.path.exists(pkl) and os.path.exists(meta):
        MODEL = load(pkl)
        with open(meta,"r",encoding="utf-8") as f:
            FEATURE_META = json.load(f)
try_load_model()

def extract_features_for_ml(bgr, fruit):
    """Harus konsisten dengan fitur saat training (hist H + mean/std S,V + one-hot buah)."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    _, mask = segment_fruit(bgr)
    idx = mask>0
    if idx.sum()==0:
        idx = np.ones(mask.shape, dtype=bool)
    H = hsv[:,:,0][idx].astype(np.float32)
    S = hsv[:,:,1][idx].astype(np.float32)
    V = hsv[:,:,2][idx].astype(np.float32)
    bins = FEATURE_META["bins"] if FEATURE_META else 16
    hist,_ = np.histogram(H, bins=bins, range=(0,180), density=True)
    sv_mean = np.array([S.mean(), V.mean()], dtype=np.float32)
    sv_std  = np.array([S.std(ddof=1), V.std(ddof=1)], dtype=np.float32)
    fruits = FEATURE_META["fruits"] if FEATURE_META else ["banana","orange","apple"]
    oh = np.zeros(len(fruits), dtype=np.float32)
    if fruit in fruits:
        oh[fruits.index(fruit)] = 1.0
    feat = np.hstack([hist, sv_mean, sv_std, oh]).reshape(1,-1)
    return feat

class AnalyzeOut(BaseModel):
    fruit: str
    label: str
    score: float
    p_ripe: float
    p_unripe: float
    p_over: float
    overlay_data_url: str | None = None

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/status")
def status():
    """Cek apakah model ML sudah terbaca."""
    return {
        "ok": True,
        "model_loaded": MODEL is not None,
        "feature_meta": FEATURE_META if FEATURE_META else None
    }

@app.post("/analyze", response_model=AnalyzeOut)
async def analyze(
    file: UploadFile = File(...),
    fruit: str = Form(...),
    mode: str = Form("rule")  # <-- "rule" (default) atau "ml"
):
    fruit = fruit.lower()
    if fruit not in ("banana","orange","apple"):
        fruit = "banana"

    bgr = read_image_to_bgr(file)
    hsv, mask = segment_fruit(bgr)

    # ====== MODE: ML ======
    if mode == "ml":
        if MODEL is None or FEATURE_META is None:
            raise HTTPException(
                status_code=400,
                detail="Model belum tersedia. Latih dulu (train_rf.py) atau set mode=rule."
            )
        feat = extract_features_for_ml(bgr, fruit)
        pred = MODEL.predict(feat)[0]  # "unripe"/"ripe"/"overripe"

        # Skor dan probabilitas (jika model mendukung)
        score = 0.0
        p_unripe = p_ripe = p_over = 0.0
        if hasattr(MODEL, "predict_proba"):
            labels = FEATURE_META.get("labels", ["unripe","ripe","overripe"])
            proba = MODEL.predict_proba(feat)[0]
            def getp(name):
                return float(proba[labels.index(name)]) if name in labels else 0.0
            p_unripe, p_ripe, p_over = getp("unripe"), getp("ripe"), getp("overripe")
            score = p_ripe  # gunakan prob. ripe sebagai skor kematangan

        label = pred
        overlay = make_overlay(bgr, mask, label)
        return AnalyzeOut(
            fruit=fruit, label=label, score=score,
            p_ripe=p_ripe, p_unripe=p_unripe, p_over=p_over,
            overlay_data_url=overlay
        )

    # ====== MODE: RULE (default) ======
    label, stats = classify(hsv, mask, fruit)
    overlay = make_overlay(bgr, mask, label)
    return AnalyzeOut(fruit=fruit, label=label, overlay_data_url=overlay, **stats)