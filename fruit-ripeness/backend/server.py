from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np, cv2, base64

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

class AnalyzeOut(BaseModel):
    fruit: str
    label: str
    score: float
    p_ripe: float
    p_unripe: float
    p_over: float
    overlay_data_url: str | None = None

@app.get("/health")
def health(): return {"ok": True}

@app.post("/analyze", response_model=AnalyzeOut)
async def analyze(file: UploadFile = File(...), fruit: str = Form(...)):
    fruit = fruit.lower()
    if fruit not in ("banana","orange","apple"): fruit = "banana"
    bgr = read_image_to_bgr(file)
    hsv, mask = segment_fruit(bgr)
    label, stats = classify(hsv, mask, fruit)
    overlay = make_overlay(bgr, mask, label)
    return AnalyzeOut(fruit=fruit, label=label, overlay_data_url=overlay, **stats)