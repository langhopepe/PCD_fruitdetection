# server_onnx.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
import os, io, json
import numpy as np
from PIL import Image
import onnxruntime as ort

# -----------------------------
# Konfigurasi
# -----------------------------
HERE = Path(__file__).parent.resolve()
MODELS_DIR = HERE / "models"

# Threshold untuk menolak non-target di GATE (0..1)
GATE_MIN_CONF = float(os.getenv("GATE_MIN_CONF", "0.80"))

# Normalisasi (sesuai training ResNet18 torchvision weights)
IMG_SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Provider ONNX Runtime (CPU by default)
# Jika kamu pakai onnxruntime-gpu dan ingin CUDA: set env ORT_PROVIDER=CUDAExecutionProvider
PROVIDERS = [os.getenv("ORT_PROVIDER", "CPUExecutionProvider")]

# -----------------------------
# Util gambar: resize -> centercrop -> to NCHW float32 -> normalize
# -----------------------------
def preprocess_pil(pil: Image.Image) -> np.ndarray:
    pil = pil.convert("RGB")
    w, h = pil.size
    # resize: short side -> 256 (seperti eval pipeline)
    short, long = (w, h) if w < h else (h, w)
    scale = 256.0 / short
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    pil = pil.resize((new_w, new_h), Image.BILINEAR)
    # center crop 224x224
    left = (pil.width - IMG_SIZE) // 2
    top  = (pil.height - IMG_SIZE) // 2
    pil = pil.crop((left, top, left + IMG_SIZE, top + IMG_SIZE))
    # to tensor (HWC->CHW), normalize
    x = np.asarray(pil).astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    x = np.transpose(x, (2, 0, 1))  # CHW
    x = x[None, ...]  # NCHW
    return x

def softmax(logits: np.ndarray) -> np.ndarray:
    # logits: (N, C)
    x = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)

# -----------------------------
# Load sessions & labels
# -----------------------------
def load_labels(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def make_session(onnx_path: Path) -> ort.InferenceSession:
    if not onnx_path.exists():
        raise FileNotFoundError(f"Model not found: {onnx_path}")
    try:
        return ort.InferenceSession(str(onnx_path), providers=PROVIDERS)
    except Exception as e:
        # fallback ke CPU jika provider gagal
        return ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

# Gate (4 kelas: apple/banana/orange/other)
GATE_ONNX  = MODELS_DIR / "gate_resnet18.onnx"
GATE_LABEL = MODELS_DIR / "gate_resnet18.labels.json"
gate_sess  = make_session(GATE_ONNX)
gate_classes = load_labels(GATE_LABEL)

# Ripeness per buah (3 kelas)
RIPEN_SESS = {}
RIPEN_CLS  = {}
for fruit in ["apple", "banana", "orange"]:
    onnx_p = MODELS_DIR / f"ripeness_{fruit}_resnet18.onnx"
    lbl_p  = MODELS_DIR / f"ripeness_{fruit}_resnet18.labels.json"
    RIPEN_SESS[fruit] = make_session(onnx_p)
    RIPEN_CLS[fruit]  = load_labels(lbl_p)

# -----------------------------
# FastAPI
# -----------------------------
app = FastAPI(title="Fruit Ripeness ONNX API", version="1.0")

# CORS: sesuaikan dengan FE dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalyzeOut(BaseModel):
    fruit: str
    fruit_conf: float
    label: str
    score: float
    probs: dict
    overlay_data_url: str | None = None  # biarkan untuk kompatibilitas FE

@app.get("/health")
def health():
    return {"ok": True, "provider": gate_sess.get_providers(), "gate_min_conf": GATE_MIN_CONF}

# -----------------------------
# /analyze
# -----------------------------
@app.post("/analyze", response_model=AnalyzeOut)
async def analyze(file: UploadFile = File(...)):
    # 1) Read image
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="File kosong.")
    try:
        pil = Image.open(io.BytesIO(raw))
    except Exception:
        raise HTTPException(status_code=400, detail="File bukan gambar yang valid.")

    x = preprocess_pil(pil)  # (1,3,224,224)

    # 2) Gate: apple/banana/orange/other
    g_logits = gate_sess.run(None, {"input": x})[0]  # (1,4)
    g_probs  = softmax(g_logits)[0]
    g_idx    = int(np.argmax(g_probs))
    g_label  = gate_classes[g_idx]
    g_conf   = float(g_probs[g_idx])

    # Jika gate memprediksi "other" atau confidence di bawah threshold → tolak
    if g_label not in ("apple", "banana", "orange") or g_conf < GATE_MIN_CONF:
        # Sertakan info kelas terdekat untuk memudahkan user
        raise HTTPException(
            status_code=422,
            detail={
                "error": "Gambar tidak dikenali sebagai apel/pisang/jeruk.",
                "predicted": g_label,
                "confidence": round(g_conf, 4),
                "hint": "Upload foto apple, banana, atau orange dengan objek buah dominan."
            }
        )

    fruit = g_label

    # 3) Ripeness untuk buah terpilih
    r_logits = RIPEN_SESS[fruit].run(None, {"input": x})[0]  # (1,3)
    r_probs  = softmax(r_logits)[0]
    r_idx    = int(np.argmax(r_probs))
    r_label  = RIPEN_CLS[fruit][r_idx]
    r_conf   = float(r_probs[r_idx])

    # Normalisasi key probs agar FE kamu selalu punya: unripe / ripe / overripe
    # (urutan di labels.json bisa berbeda-beda)
    probs_map = {k: 0.0 for k in ["unripe", "ripe", "overripe"]}
    for i, name in enumerate(RIPEN_CLS[fruit]):
        if name in probs_map:
            probs_map[name] = float(r_probs[i])

    return AnalyzeOut(
        fruit=fruit,
        fruit_conf=g_conf,
        label=r_label,
        score=r_conf,
        probs=probs_map,
        overlay_data_url=None
    )
