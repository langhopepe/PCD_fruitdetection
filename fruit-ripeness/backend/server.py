# server.py  — stable color-only, no scikit-image required
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np, cv2, base64, os, json, io, traceback
from joblib import load

# -------- CORS (dev) --------
ALLOWED_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]

app = FastAPI(title="Fruit Ripeness API", version="3.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=86400,
)

# -------- Optional Pillow (fallback WebP) --------
try:
    from PIL import Image
    HAS_PIL = True
except Exception:
    HAS_PIL = False
    Image = None

# -------- Model / meta --------
MODEL = None
FEATURE_META = {"resize_width": 256, "hsv_bins": 32}  # HARUS match saat training
MODEL_INFO = {"path": None, "n_features_in": None}

def load_model_and_meta():
    """Load model & meta dari backend/models"""
    global MODEL, FEATURE_META, MODEL_INFO
    here = os.path.dirname(__file__)
    p_model = os.path.join(here, "models", "model_ml.pkl")
    p_meta  = os.path.join(here, "models", "feature_meta.json")

    if os.path.exists(p_model):
        MODEL = load(p_model)
        MODEL_INFO["path"] = p_model
        MODEL_INFO["n_features_in"] = getattr(MODEL, "n_features_in_", None)

    if os.path.exists(p_meta):
        try:
            with open(p_meta, "r", encoding="utf-8") as f:
                meta = json.load(f)
                FEATURE_META["resize_width"] = meta.get("resize_width", FEATURE_META["resize_width"])
                FEATURE_META["hsv_bins"] = meta.get("hsv_bins", FEATURE_META["hsv_bins"])
        except Exception:
            pass

load_model_and_meta()

def expected_feature_dim(meta: dict) -> int:
    """color-only: hist(H) bins + mean(S,V) + std(S,V) = bins + 4"""
    return int(meta.get("hsv_bins", 32) + 4)

# -------- IO utils --------
def read_bytes_to_bgr(raw: bytes, resize_width: int = 256):
    """Decode -> BGR. Coba OpenCV; fallback Pillow (WebP). Resize by width."""
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

# -------- Fitur: COLOR ONLY (tanpa LBP) --------
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

# -------- Precheck: hanya apple/banana/orange --------
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
    if area < 0.0025 * (h_img * w_img):  # ambang dilonggarkan
        return "unknown"

    x, y, w, h = cv2.boundingRect(c)
    ar = max(w, h) / max(1, min(w, h))

    idx = np.zeros_like(mask, dtype=np.uint8)
    cv2.drawContours(idx, [c], -1, 255, thickness=-1)
    idx_bool = idx > 0

    H = hsv[:, :, 0][idx_bool]
    S = hsv[:, :, 1][idx_bool]
    if H.size < 300:
        return "unknown"

    h_med = float(np.median(H))
    s_med = float(np.median(S))

    if ar > 1.4 and 18 <= h_med <= 45:  # banana (kuning, memanjang)
        return "banana"
    if 6 <= h_med <= 28 and s_med >= 70:  # orange
        return "orange"
    if (h_med <= 15 or h_med >= 165) or (32 <= h_med <= 90):  # apple (merah/hijau)
        return "apple"
    return "unknown"

# -------- Schemas --------
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
        "model_info": MODEL_INFO,
        "expected_feature_dim": expected_feature_dim(FEATURE_META),
    }

# -------- Endpoint utama --------
@app.post("/analyze", response_model=AnalyzeOut)
async def analyze(
    file: UploadFile = File(...),
    force: int | None = Form(None)  # kirim force=1 utk bypass precheck (opsional)
):
    if MODEL is None:
        raise HTTPException(status_code=500, detail="Model belum tersedia. Jalankan training dulu.")

    try:
        raw = await file.read()
        bgr = read_bytes_to_bgr(raw, FEATURE_META.get("resize_width", 256))
        if bgr is None:
            raise HTTPException(status_code=400, detail="File bukan gambar valid (decode gagal).")

        if not force:
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

        feat = features_from_bgr_color_only(bgr, FEATURE_META).reshape(1, -1)

        # --- Diagnostik jumlah fitur (agar 500-nya jelas kalau mismatch) ---
        exp = expected_feature_dim(FEATURE_META)
        if feat.shape[1] != exp:
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "feature_dim_mismatch",
                    "feat_dim_runtime": int(feat.shape[1]),
                    "feat_dim_expected": int(exp),
                    "hint": "Retrain dengan train_ml_color_only.py memakai FEATURE_META yang sama (hsv_bins/resize_width) atau samakan kode server.",
                },
            )
        if MODEL_INFO["n_features_in"] is not None and MODEL_INFO["n_features_in"] != feat.shape[1]:
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "model_n_features_in_mismatch",
                    "model_n_features_in": int(MODEL_INFO["n_features_in"]),
                    "feat_dim_runtime": int(feat.shape[1]),
                    "hint": "Model dilatih dg jumlah fitur berbeda. Latih ulang (color-only) atau sesuaikan server.",
                },
            )

        # Prediksi
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

        # Overlay ringkas
        overlay = bgr.copy()
        cv2.rectangle(overlay, (10, 10), (320, 70), (0, 0, 0), -1)
        cv2.putText(overlay, f"{str(yhat).upper()} ({score:.2f})", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        ok, buf = cv2.imencode(".png", overlay)
        overlay_url = "data:image/png;base64," + base64.b64encode(buf).decode("utf-8") if ok else None

        return AnalyzeOut(label=str(yhat), score=float(score), probs=probs, overlay_data_url=overlay_url)

    except HTTPException:
        raise
    except Exception as e:
        print("ANALYZE_ERROR:", repr(e))
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal error saat memproses gambar.")
