# mmseg/datasets/transforms/multilabel_transforms.py
import random
import cv2
import numpy as np

from mmseg.registry import TRANSFORMS
from mmcv.transforms import BaseTransform

def _resize_img(img: np.ndarray, new_w: int, new_h: int):
    interp = cv2.INTER_LINEAR
    return cv2.resize(img, (new_w, new_h), interpolation=interp)

def _resize_mask_ch(mask_ch: np.ndarray, new_w: int, new_h: int):
    # mask는 0/1이라 nearest
    return cv2.resize(mask_ch, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

@TRANSFORMS.register_module()
class RandomResizeML(BaseTransform):
    """
    mmseg RandomResize 느낌을 단순화:
    - scale=(W, H) 기준
    - ratio_range=(min,max) 중 랜덤 ratio
    - keep_ratio=True면 원본 비율 유지
    """
    def __init__(self, scale, ratio_range=(0.5, 2.0), keep_ratio=True):
        self.base_w, self.base_h = scale
        self.rmin, self.rmax = ratio_range
        self.keep_ratio = keep_ratio

    def transform(self, results: dict) -> dict:
        img = results["img"]
        gt = results.get("gt_multi_hot", None)

        h, w = img.shape[:2]
        ratio = random.uniform(self.rmin, self.rmax)

        target_w = int(self.base_w * ratio)
        target_h = int(self.base_h * ratio)

        if self.keep_ratio:
            # 원본 비율 유지: 짧은 변 기준으로 맞추기
            scale = min(target_w / w, target_h / h)
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
        else:
            new_w, new_h = target_w, target_h

        # img
        img2 = _resize_img(img, new_w, new_h)

        # gt (C,H,W)
        if gt is not None:
            C = gt.shape[0]
            gt2 = np.zeros((C, new_h, new_w), dtype=gt.dtype)
            for c in range(C):
                gt2[c] = _resize_mask_ch(gt[c], new_w, new_h)
            results["gt_multi_hot"] = gt2

        results["img"] = img2
        results["img_shape"] = img2.shape[:2]
        return results


@TRANSFORMS.register_module()
class RandomCropML(BaseTransform):
    def __init__(self, crop_size=(512, 512)):
        self.crop_h, self.crop_w = crop_size

    def transform(self, results: dict) -> dict:
        img = results["img"]
        gt = results.get("gt_multi_hot", None)

        h, w = img.shape[:2]
        ch, cw = self.crop_h, self.crop_w

        # 이미지가 crop보다 작으면 패딩 (간단 pad)
        pad_h = max(0, ch - h)
        pad_w = max(0, cw - w)
        if pad_h > 0 or pad_w > 0:
            img = np.pad(img, ((0, pad_h), (0, pad_w)) if img.ndim == 2 else ((0, pad_h), (0, pad_w), (0, 0)),
                         mode="constant", constant_values=0)
            if gt is not None:
                gt = np.pad(gt, ((0, 0), (0, pad_h), (0, pad_w)),
                            mode="constant", constant_values=0)
            h, w = img.shape[:2]

        top = random.randint(0, h - ch)
        left = random.randint(0, w - cw)

        img2 = img[top:top+ch, left:left+cw]

        if gt is not None:
            gt2 = gt[:, top:top+ch, left:left+cw]
            results["gt_multi_hot"] = gt2

        results["img"] = img2
        results["img_shape"] = img2.shape[:2]
        return results


@TRANSFORMS.register_module()
class RandomFlipML(BaseTransform):
    def __init__(self, prob=0.5, direction="horizontal"):
        self.prob = prob
        self.direction = direction

    def transform(self, results: dict) -> dict:
        if random.random() > self.prob:
            return results

        img = results["img"]
        gt = results.get("gt_multi_hot", None)

        if self.direction == "horizontal":
            img2 = np.flip(img, axis=1)
            if gt is not None:
                gt2 = np.flip(gt, axis=2)
                results["gt_multi_hot"] = gt2
        elif self.direction == "vertical":
            img2 = np.flip(img, axis=0)
            if gt is not None:
                gt2 = np.flip(gt, axis=1)
                results["gt_multi_hot"] = gt2
        else:
            raise ValueError("direction must be 'horizontal' or 'vertical'")

        results["img"] = img2.copy()
        results["img_shape"] = img2.shape[:2]
        return results
