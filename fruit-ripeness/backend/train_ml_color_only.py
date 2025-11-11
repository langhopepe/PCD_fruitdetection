# train_ml_color_only.py
import sys, os, json, io, glob
import numpy as np, cv2
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# --- fallback Pillow utk WEBP ---
try:
    from PIL import Image
    HAS_PIL = True
except Exception:
    HAS_PIL = False
    Image = None

FEATURE_META = {"resize_width": 256, "hsv_bins": 32}  # harus sama dg server_debug.py
ALLOWED_FRUITS = {"banana", "orange", "apple"}
ALLOWED_RIPENESS = {"unripe", "ripe", "overripe"}

def read_path_to_bgr(path, resize_width=256):
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None and HAS_PIL:
        try:
            pil = Image.open(path).convert("RGB")
            img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        except Exception:
            return None
    if img is None:
        return None
    h, w = img.shape[:2]
    if w and w != resize_width:
        scale = resize_width / w
        img = cv2.resize(img, (resize_width, int(h * scale)))
    return img

def features_from_bgr_color_only(bgr, meta):
    bins = meta.get("hsv_bins", 32)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H = hsv[:, :, 0].astype(np.float32).ravel()
    S = hsv[:, :, 1].astype(np.float32).ravel()
    V = hsv[:, :, 2].astype(np.float32).ravel()
    hist_h, _ = np.histogram(H, bins=bins, range=(0, 180), density=True)
    sv_mean = np.array([S.mean(), V.mean()], dtype=np.float32)
    sv_std  = np.array([S.std(ddof=1), V.std(ddof=1)], dtype=np.float32)
    return np.hstack([hist_h, sv_mean, sv_std]).astype(np.float32)  # dim = bins + 4

def scan_dataset(root):
    X, y = [], []
    missing = 0
    for fruit in sorted(os.listdir(root)):
        fruit_path = os.path.join(root, fruit)
        if not os.path.isdir(fruit_path): continue
        if fruit.lower() not in ALLOWED_FRUITS: continue
        for ripeness in sorted(os.listdir(fruit_path)):
            ripeness_path = os.path.join(fruit_path, ripeness)
            if not os.path.isdir(ripeness_path): continue
            if ripeness.lower() not in ALLOWED_RIPENESS: continue
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp"):
                for fp in glob.glob(os.path.join(ripeness_path, ext)):
                    bgr = read_path_to_bgr(fp, FEATURE_META["resize_width"])
                    if bgr is None:
                        missing += 1
                        continue
                    feat = features_from_bgr_color_only(bgr, FEATURE_META)
                    X.append(feat)
                    y.append(ripeness.lower())
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=object)
    print(f"[scan] X shape={X.shape}, y={len(y)}, missing/failed={missing}")
    return X, y

def resolve_dataset_dir():
    # 1) CLI arg
    if len(sys.argv) > 1:
        p = sys.argv[1]
        if os.path.isdir(p):
            return os.path.abspath(p)
        raise SystemExit(f"[ERR] Argumen path dataset tidak ditemukan: {p}")
    # 2) ENV
    envp = os.environ.get("FRUIT_DATASET_DIR")
    if envp and os.path.isdir(envp):
        return os.path.abspath(envp)
    # 3) Default ke ../dataset (satu level di luar backend)
    here = os.path.dirname(__file__)
    default = os.path.abspath(os.path.join(here, "..", "dataset"))
    if os.path.isdir(default):
        return default
    raise SystemExit("[ERR] Dataset tidak ditemukan. Beri argumen path atau set FRUIT_DATASET_DIR.")


if __name__ == "__main__":
    DATASET_DIR = resolve_dataset_dir()
    print(f"[info] Using dataset dir: {DATASET_DIR}")
    X, y = scan_dataset(DATASET_DIR)
    if len(y) == 0:
        raise SystemExit("Dataset kosong/format tidak sesuai. Harus: dataset/<fruit>/<ripeness>/*.jpg")

    # split
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print({cls: int((y == cls).sum()) for cls in sorted(set(y))})

    # model
    clf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced_subsample")
    clf.fit(Xtr, ytr)
    print("train acc:", clf.score(Xtr, ytr))
    print("test  acc:", clf.score(Xte, yte))

    # simpan
    here = os.path.dirname(__file__)
    os.makedirs(os.path.join(here, "models"), exist_ok=True)
    dump(clf, os.path.join(here, "models", "model_ml.pkl"))
    with open(os.path.join(here, "models", "feature_meta.json"), "w", encoding="utf-8") as f:
        json.dump(FEATURE_META, f, ensure_ascii=False, indent=2)

    print("[done] saved to models/model_ml.pkl & models/feature_meta.json")
    print("[classes]", sorted(set(y)))  # setelah scan_dataset()
