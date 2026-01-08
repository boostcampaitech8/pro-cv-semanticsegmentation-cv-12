# 0_make_masks_and_split_ml.py
import json
import random
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

CLASSES = [
    'finger-1','finger-2','finger-3','finger-4','finger-5',
    'finger-6','finger-7','finger-8','finger-9','finger-10',
    'finger-11','finger-12','finger-13','finger-14','finger-15',
    'finger-16','finger-17','finger-18','finger-19',
    'Trapezium','Trapezoid','Capitate','Hamate','Scaphoid','Lunate',
    'Triquetrum','Pisiform','Radius','Ulna'
]
CLASS2IDX = {c: i for i, c in enumerate(CLASSES)}

ROOT = Path("/data/ephemeral/home")
DATA = ROOT / "data"
TRAIN_IMG = DATA / "train" / "DCM"
TRAIN_JSON = DATA / "train" / "outputs_json"

MASK_DIR = DATA / "masks_train_ml"     # ✅ npz 저장 폴더 (train/val 따로 안 나눔)
TRAIN_LIST = DATA / "train_list_ml.txt"
VAL_LIST = DATA / "val_list_ml.txt"

SEED = 42
VAL_RATIO = 0.2  

def json_path_for(img_path: Path) -> Path:
    # 기존 단일라벨 코드와 동일한 매핑 방식 :contentReference[oaicite:2]{index=2}
    rel = img_path.relative_to(TRAIN_IMG)  # IDxxx/filename.png
    return TRAIN_JSON / rel.with_suffix(".json")

def build_multihot_mask_from_json(json_path: Path, h: int, w: int) -> np.ndarray:
    """(29,H,W) uint8 multi-hot mask"""
    gt = np.zeros((len(CLASSES), h, w), dtype=np.uint8)

    if not json_path.exists():
        return gt

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for ann in data.get("annotations", []):
        if ann.get("type") != "poly_seg":
            continue

        label = ann.get("label")
        if label not in CLASS2IDX:
            continue

        pts = np.array(ann.get("points", []), dtype=np.int32)
        if pts.ndim != 2 or pts.shape[0] < 3:
            continue

        c = CLASS2IDX[label]
        tmp = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(tmp, [pts], 1)         # ✅ 이 클래스 채널만 1
        gt[c] = np.maximum(gt[c], tmp)      # ✅ 겹침 허용

    return gt

def main():
    random.seed(SEED)

    imgs = sorted(TRAIN_IMG.rglob("*.png"))
    random.shuffle(imgs)

    n_val = int(len(imgs) * VAL_RATIO)
    val_set = set(imgs[:n_val])
    train_set = imgs[n_val:]

    MASK_DIR.mkdir(parents=True, exist_ok=True)

    for img_path in tqdm(imgs, desc="make multi-hot npz"):
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"cannot read: {img_path}")
        h, w = img.shape

        gt = build_multihot_mask_from_json(json_path_for(img_path), h, w)

        # ✅ 폴더 구조(IDxxx/...) 유지해서 저장 (권장)
        rel = img_path.relative_to(TRAIN_IMG)   # IDxxx/filename.png
        out_dir = MASK_DIR / rel.parent
        out_dir.mkdir(parents=True, exist_ok=True)

        out_path = out_dir / f"{rel.stem}.npz"
        np.savez_compressed(out_path, mask=gt)

    # ✅ 리스트에는 "상대경로(IDxxx/filename.png)" 저장 (단일라벨 코드와 동일 패턴) :contentReference[oaicite:3]{index=3}
    with open(TRAIN_LIST, "w", encoding="utf-8") as f:
        for p in sorted(train_set):
            f.write(str(p.relative_to(TRAIN_IMG)).replace("\\", "/") + "\n")

    with open(VAL_LIST, "w", encoding="utf-8") as f:
        for p in sorted(val_set):
            f.write(str(p.relative_to(TRAIN_IMG)).replace("\\", "/") + "\n")

    print("DONE")
    print("mask dir:", MASK_DIR)
    print("train list:", TRAIN_LIST)
    print("val list:", VAL_LIST)

if __name__ == "__main__":
    main()
