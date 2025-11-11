# server_debug.py — SAFE MODE
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np, cv2, base64, os, json, io, traceback
from joblib import load

# --- CORS (dev) ---
ALLOWED_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]
app = FastAPI(title="Fruit Ripeness API [SAFE MODE]", version="D1")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=86400,
)

# --- MODEL ---
MODEL = None
FEATURE_META = {"resize_width": 256, "hsv_bins": 32}  # LBP sengaja dimatikan

def load_model_and_meta():
    global MODEL, FEATURE_META
    here = os.path.dirname(__file__)
    p_model = os.path.join(here, "models", "model_ml.pkl")
    p_meta  = os.path.join(here, "models", "feature_meta.json")
    if not os.path.exists(p_model):
        return False
    MODEL = load(p_model)
    if os.path.exists(p_meta):
        try:
            with open(p_meta, "r", encoding="utf-8") as f:
                meta = json.load(f)
                FEATURE_META["resize_width"] = meta.get("resize_width", 256)
                FEATURE_META["hsv_bins"] = meta.get("hsv_bins", 32)
        except Exception:
            pass
    return True

_ = load_model_and_meta()

# --- IO utils (OpenCV + fallback Pillow untuk WEBP) ---
try:
    from PIL import Image
    HAS_PIL = True
except Exception:
    HAS_PIL = False
    Image = None

def read_bytes_to_bgr(raw: bytes, resize_width: int = 256):
    arr = np.frombuffer(raw, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None and HAS_PIL:
        try:
            pil = Image.open(io.BytesIO(raw)).convert("RGB")
            img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        except Exception:
            img = None
    if img is None:
        return None
    h, w = img.shape[:2]
    if w and w != resize_width:
        scale = resize_width / w
        img = cv2.resize(img, (resize_width, int(h * scale)))
    return img

# --- fitur: hanya warna (tanpa LBP) ---
def features_from_bgr_color_only(bgr, meta):
    bins = meta.get("hsv_bins", 32)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H = hsv[:, :, 0].astype(np.float32).ravel()
    S = hsv[:, :, 1].astype(np.float32).ravel()
    V = hsv[:, :, 2].astype(np.float32).ravel()

    hist_h, _ = np.histogram(H, bins=bins, range=(0, 180), density=True)
    sv_mean = np.array([S.mean(), V.mean()], dtype=np.float32)
    sv_std  = np.array([S.std(ddof=1), V.std(ddof=1)], dtype=np.float32)

    feat = np.hstack([hist_h, sv_mean, sv_std]).astype(np.float32)  # dim = bins + 4
    return feat

# --- precheck sederhana: tolak non-target (apple/banana/orange) ---
ALLOWED_FRUITS = ["banana", "orange", "apple"]

def _segment_color_mask(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 40, 40), (180, 255, 255))
    k = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, 1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, 2)
    return hsv, mask

def guess_fruit(bgr):
    hsv, mask = _segment_color_mask(bgr)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return "unknown"
    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    h_img, w_img = mask.shape
    if area < 0.01 * (h_img * w_img):
        return "unknown"
    x, y, w, h = cv2.boundingRect(c)
    ar = max(w, h) / max(1, min(w, h))
    idx = np.zeros_like(mask, dtype=np.uint8)
    cv2.drawContours(idx, [c], -1, 255, thickness=-1)
    idx_bool = idx > 0
    H = hsv[:, :, 0][idx_bool]
    S = hsv[:, :, 1][idx_bool]
    if H.size < 500:
        return "unknown"
    h_med = float(np.median(H))
    s_med = float(np.median(S))
    if ar > 1.6 and 20 <= h_med <= 40: return "banana"
    if 8 <= h_med <= 25 and s_med >= 90: return "orange"
    if (h_med <= 10 or h_med >= 170) or (35 <= h_med <= 85): return "apple"
    return "unknown"

# --- schema & endpoints ---
class AnalyzeOut(BaseModel):
    label: str
    score: float
    probs: dict | None = None
    overlay_data_url: str | None = None

@app.get("/health")
def health():
    return {
        "ok": True,
        "model_loaded": MODEL is not None,
        "feature_meta": FEATURE_META,
        "safe_mode": True,
        "has_pillow": HAS_PIL,
    }

@app.post("/analyze", response_model=AnalyzeOut)
async def analyze(file: UploadFile = File(...)):
    if MODEL is None:
        raise HTTPException(status_code=500, detail="Model belum tersedia. Jalankan training dulu.")

    try:
        # --- Step 1: decode ---
        raw = await file.read()
        bgr = read_bytes_to_bgr(raw, FEATURE_META.get("resize_width", 256))
        if bgr is None:
            raise HTTPException(status_code=400, detail="File bukan gambar valid (decode gagal).")

        # --- Step 2: precheck ---
        guessed = guess_fruit(bgr)
        if guessed not in ALLOWED_FRUITS:
            import random
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "Gambar tidak dikenali sebagai apel/pisang/jeruk.",
                    "suggested_fruit": random.choice(ALLOWED_FRUITS),
                },
            )

        # --- Step 3: features (warna saja, aman) ---
        feat = features_from_bgr_color_only(bgr, FEATURE_META).reshape(1, -1)

        # --- Step 4: predict (robust) ---
        yhat = MODEL.predict(feat)[0]
        score = 0.0
        probs = None
        if hasattr(MODEL, "predict_proba"):
            try:
                proba = MODEL.predict_proba(feat)
                vec = np.asarray(proba)[0]
                classes = getattr(MODEL, "classes_", None)
                if classes is not None and len(classes) == len(vec):
                    probs = {str(classes[i]): float(vec[i]) for i in range(len(vec))}
                else:
                    probs = {f"class_{i}": float(vec[i]) for i in range(len(vec))}
                score = float(probs.get("ripe", max(probs.values())))
            except Exception:
                score = 1.0 if str(yhat).lower() == "ripe" else 0.0

        # --- Step 5: overlay ---
        overlay = bgr.copy()
        cv2.rectangle(overlay, (10, 10), (280, 70), (0, 0, 0), -1)
        cv2.putText(overlay, f"{str(yhat).upper()} ({score:.2f})", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        ok, buf = cv2.imencode(".png", overlay)
        overlay_url = "data:image/png;base64," + base64.b64encode(buf).decode("utf-8") if ok else None

        return AnalyzeOut(label=str(yhat), score=score, probs=probs, overlay_data_url=overlay_url)

    except HTTPException:
        raise
    except Exception as e:
        print("ANALYZE_ERROR:", repr(e))
        traceback.print_exc()
        # kirim pesan detail untuk debug FE (jangan dipakai di production)
        raise HTTPException(status_code=500, detail=f"Internal error: {type(e).__name__}: {e}")
