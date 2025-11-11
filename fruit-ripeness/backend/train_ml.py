import os, json
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from joblib import dump
from utils_features import scan_dataset, read_and_resize, features_from_bgr, DEFAULT_FEATURE_META
import cv2

LABELS = ["unripe","ripe","overripe"]

def load_features(dataset_root, meta):
    paths, y_text = scan_dataset(dataset_root, LABELS)
    if not paths:
        raise SystemExit(f"Tidak menemukan gambar di: {dataset_root}")
    X, y = [], []
    for p, lab in zip(paths, y_text):
        bgr = read_and_resize(p, meta["resize_width"])
        if bgr is None: 
            continue
        feat = features_from_bgr(bgr, meta)
        X.append(feat)
        y.append(lab)
    return np.array(X, dtype=np.float32), np.array(y), paths

if __name__ == "__main__":
    here = Path(__file__).parent
    DATASET = str(here.parent / "dataset")  # ../dataset
    MODELS  = here / "models"
    MODELS.mkdir(parents=True, exist_ok=True)

    # konfigurasi fitur (bisa kamu ubah sesuai dataset)
    feature_meta = DEFAULT_FEATURE_META.copy()
    feature_meta["resize_width"] = 256
    feature_meta["hsv_bins"]     = 32
    feature_meta["lbp_radius"]   = 2
    feature_meta["lbp_method"]   = "uniform"

    print("[INFO] Load dataset & extract features ...")
    X, y, paths = load_features(DATASET, feature_meta)
    print(f"[INFO] samples={len(y)}, dim={X.shape[1]}")

    # split train/val/test: 70/15/15
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)
    X_va, X_te, y_va, y_te   = train_test_split(X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=42)

    print("[INFO] Train RandomForest ...")
    clf = RandomForestClassifier(
        n_estimators=400, max_depth=None, n_jobs=-1, random_state=42, class_weight="balanced"
    )
    clf.fit(X_tr, y_tr)

    print("\n[VAL] confusion matrix:")
    print(confusion_matrix(y_va, clf.predict(X_va), labels=LABELS))
    print(classification_report(y_va, clf.predict(X_va), labels=LABELS, digits=4))

    print("\n[TEST] confusion matrix:")
    print(confusion_matrix(y_te, clf.predict(X_te), labels=LABELS))
    print(classification_report(y_te, clf.predict(X_te), labels=LABELS, digits=4))

    # simpan
    from joblib import dump
    dump(clf, MODELS/"model_ml.pkl")
    with open(MODELS/"feature_meta.json","w",encoding="utf-8") as f:
        json.dump(feature_meta, f, ensure_ascii=False, indent=2)
    with open(MODELS/"labels.json","w",encoding="utf-8") as f:
        json.dump(LABELS, f, ensure_ascii=False, indent=2)

    print("\n[SAVED] models/model_ml.pkl + feature_meta.json + labels.json")
