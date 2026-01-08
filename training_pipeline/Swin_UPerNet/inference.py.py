# tools_local/2_infer_upernet_swin_base_ml_2048_tta_hflip_whole.py
from pathlib import Path
import uuid
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import cv2

from mmengine.config import Config
from mmseg.apis import init_model, inference_model

CLASSES = [
    'finger-1','finger-2','finger-3','finger-4','finger-5',
    'finger-6','finger-7','finger-8','finger-9','finger-10',
    'finger-11','finger-12','finger-13','finger-14','finger-15',
    'finger-16','finger-17','finger-18','finger-19',
    'Trapezium','Trapezoid','Capitate','Hamate','Scaphoid','Lunate',
    'Triquetrum','Pisiform','Radius','Ulna'
]

def rle_encode(mask: np.ndarray) -> str:
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(map(str, runs))

def build_cfg_whole_noresize(cfg_path: str):
    """
    - Whole inference 강제: model.test_cfg = mode='whole'
    - Resize 제거: test_pipeline을 LoadImageFromFile + PackSegInputsML 로 override
      => '원본 크기 그대로' 들어가고, 보통 결과도 ori_shape로 맞춰서 나옴
    """
    cfg = Config.fromfile(cfg_path)

    # 1) whole
    if isinstance(cfg.model, dict):
        cfg.model["test_cfg"] = dict(mode="whole")
    else:
        cfg.model.test_cfg = dict(mode="whole")

    # 2) no-resize pipeline
    noresize_pipeline = [
        dict(type="LoadImageFromFile"),
        dict(type="PackSegInputsML"),
    ]

    # test_dataloader가 있으면 우선 변경 (있을 수도/없을 수도)
    if "test_dataloader" in cfg:
        try:
            cfg.test_dataloader.dataset.pipeline = noresize_pipeline
        except Exception:
            pass

    # inference_model이 참조하는 test_pipeline도 같이 세팅
    cfg.test_pipeline = noresize_pipeline
    return cfg

@torch.no_grad()
def infer_whole_hflip_prob(model, img_path: Path, use_amp=True):
    """
    whole 1회 + hflip 1회 => prob 평균
    - flip 이미지는 임시 png로 저장해서 inference_model에 '파일 경로'로 넣음
      (noresize pipeline이 LoadImageFromFile 기반이기 때문에 이 방식이 안전)
    """
    # 원본 이미지 읽기 (grayscale)
    img0 = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img0 is None:
        raise FileNotFoundError(img_path)
    H, W = img0.shape[:2]

    # 1) 원본 추론
    with torch.cuda.amp.autocast(enabled=use_amp):
        out1 = inference_model(model, str(img_path))
    logits1 = out1.seg_logits.data
    if logits1.ndim == 4:
        logits1 = logits1.squeeze(0)  # (C,H,W)
    prob1 = torch.sigmoid(logits1).detach().cpu().numpy()

    # 2) hflip 이미지 만들어 임시 파일로 저장
    flip = cv2.flip(img0, 1)  # horizontal flip
    tmp_name = f"/tmp/tmp_hflip_{uuid.uuid4().hex}.png"
    cv2.imwrite(tmp_name, flip)

    try:
        with torch.cuda.amp.autocast(enabled=use_amp):
            out2 = inference_model(model, tmp_name)
        logits2 = out2.seg_logits.data
        if logits2.ndim == 4:
            logits2 = logits2.squeeze(0)
        prob2 = torch.sigmoid(logits2).detach().cpu().numpy()

        # prob2는 "뒤집힌 입력" 기준 결과이므로 다시 되돌려야 함
        prob2 = prob2[:, :, ::-1]  # (C,H,W)에서 W축 reverse

    finally:
        # 임시 파일 삭제
        try:
            Path(tmp_name).unlink(missing_ok=True)
        except Exception:
            pass

    # 안전장치: 크기 확인
    if tuple(prob1.shape[-2:]) != (H, W):
        raise RuntimeError(f"prob1 size mismatch: {prob1.shape[-2:]} vs {(H,W)} for {img_path.name}")
    if tuple(prob2.shape[-2:]) != (H, W):
        raise RuntimeError(f"prob2 size mismatch: {prob2.shape[-2:]} vs {(H,W)} for {img_path.name}")

    # 평균 앙상블
    prob = (prob1 + prob2) / 2.0
    return prob  # (C,H,W)

def main():
    mmseg_root = "/data/ephemeral/home/UPerNet/mmsegmentation"

    # Swin-B + 2048 학습 config/ckpt로 맞춰줘
    cfg_path = f"{mmseg_root}/configs/_local/upernet_swin_base_bone_ml.py"
    ckpt = f"{mmseg_root}/work_dirs/upernet_swin_base_bone_ml/best_mDice_iter_32500.pth"

    #test/DCM 하위 폴더까지 전부
    test_root = Path("/data/ephemeral/home/data/test/DCM")
    img_paths = sorted(test_root.rglob("*.png"))
    print("num test images:", len(img_paths))

    #런타임 override (재학습 필요 없음)
    cfg = build_cfg_whole_noresize(cfg_path)

    model = init_model(cfg, ckpt, device="cuda:0")
    model.eval()

    # threshold
    thr = {c: 0.5 for c in CLASSES}

    rows = []
    for img_path in tqdm(img_paths, desc="infer-whole-2048-hflip"):
        prob = infer_whole_hflip_prob(model, img_path, use_amp=True)  # (29,H,W)

        image_name = img_path.name
        for i, cls in enumerate(CLASSES):
            m = (prob[i] > thr[cls]).astype(np.uint8)
            rows.append({"image_name": image_name, "class": cls, "rle": rle_encode(m)})

    out_path = "UPerNet_swin_base_ml_2048_tta_output.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print("saved:", out_path)

if __name__ == "__main__":
    main()
