import cv2, numpy as np, os, glob
from skimage.feature import local_binary_pattern

# Konfigurasi fitur (disimpan saat training agar inference konsisten)
DEFAULT_FEATURE_META = {
    "resize_width": 256,
    "hsv_bins": 32,               # histogram Hue (0..180) → 32 bin
    "lbp_radius": 2,              # LBP tekstur
    "lbp_method": "uniform"
}

def read_and_resize(path_or_img, resize_width=256):
    if isinstance(path_or_img, str):
        bgr = cv2.imread(path_or_img)
    else:
        bgr = path_or_img
    if bgr is None:
        return None
    h, w = bgr.shape[:2]
    scale = resize_width / max(1, w)
    return cv2.resize(bgr, (resize_width, int(h*scale)))

def features_from_bgr(bgr, meta=DEFAULT_FEATURE_META):
    """Ekstraksi fitur:
       - Hist Hue (HSV) 32 bin, normalized
       - mean/std S & V
       - LBP histogram (uniform) pada gray
    """
    w = meta.get("resize_width", 256)
    bins = meta.get("hsv_bins", 32)
    lbp_radius = meta.get("lbp_radius", 2)
    lbp_method = meta.get("lbp_method", "uniform")
    n_points = 8 * lbp_radius

    # resize
    h0, w0 = bgr.shape[:2]
    if w0 != w:
        bgr = read_and_resize(bgr, resize_width=w)

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    H = hsv[:,:,0].astype(np.float32).ravel()
    S = hsv[:,:,1].astype(np.float32).ravel()
    V = hsv[:,:,2].astype(np.float32).ravel()

    # hist Hue (0..180)
    hist_h, _ = np.histogram(H, bins=bins, range=(0,180), density=True)
    sv_mean = np.array([S.mean(), V.mean()], dtype=np.float32)
    sv_std  = np.array([S.std(ddof=1), V.std(ddof=1)], dtype=np.float32)

    # LBP
    lbp = local_binary_pattern(gray, n_points, lbp_radius, method=lbp_method)
    # uniform → bins = n_points + 2
    lbp_bins = n_points + 2 if lbp_method == "uniform" else int(lbp.max()+1)
    hist_lbp, _ = np.histogram(lbp.ravel(), bins=lbp_bins, range=(0, lbp_bins), density=True)

    feat = np.hstack([hist_h, sv_mean, sv_std, hist_lbp]).astype(np.float32)
    return feat

def scan_dataset(root, labels=("unripe","ripe","overripe")):
    """Kembalikan (paths, y) dengan label diambil dari nama folder yang mengandung 'unripe|ripe|overripe'."""
    paths, y = [], []
    L = [l.lower() for l in labels]
    for ext in ("jpg","jpeg","png","bmp","webp"):
        for p in glob.glob(os.path.join(root, "**", f"*.{ext}"), recursive=True):
            parts = [pp.lower() for pp in p.split(os.sep)]
            lab = None
            for l in L:
                if any(l in part for part in parts):
                    lab = l; break
            if lab is None: 
                continue
            paths.append(p); y.append(lab)
    return paths, y
