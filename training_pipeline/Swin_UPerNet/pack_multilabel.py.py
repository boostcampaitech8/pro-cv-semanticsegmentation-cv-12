# # mmseg/datasets/transforms/pack_multilabel.py
# import numpy as np
# import torch
# from mmcv.transforms import BaseTransform 
# from mmengine.structures import PixelData
# from mmseg.structures import SegDataSample
# from mmseg.registry import TRANSFORMS

# @TRANSFORMS.register_module()
# class PackSegInputsML(BaseTransform):
#     def transform(self, results: dict) -> dict:
#         img = results["img"]
#         if img.ndim == 2:
#             img = img[None, ...]  # (1,H,W)
#         else:
#             img = img.transpose(2, 0, 1)  # (C,H,W)

#         inputs = torch.from_numpy(img).float()

#         data_sample = SegDataSample()
#         gt = results.get("gt_multi_hot", None)
#         if gt is not None:
#             gt_tensor = torch.from_numpy(gt).float()  # (29,H,W)
#             data_sample.gt_sem_seg = PixelData(data=gt_tensor)

#         results["inputs"] = inputs
#         results["data_samples"] = data_sample
#         return results

# mmseg/datasets/transforms/pack_multilabel.py
# mmseg/datasets/transforms/pack_multilabel.py

import numpy as np
import torch
from mmcv.transforms import BaseTransform
from mmengine.structures import PixelData
from mmseg.structures import SegDataSample
from mmseg.registry import TRANSFORMS

@TRANSFORMS.register_module()
class PackSegInputsML(BaseTransform):
    def transform(self, results: dict) -> dict:
        img = results["img"]
        img = np.ascontiguousarray(img)

        # ---- inputs tensor ----
        if img.ndim == 2:
            # grayscale (H,W) -> (1,H,W)
            img_chw = img[None, ...]
        else:
            # (H,W,C) -> (C,H,W)
            img_chw = img.transpose(2, 0, 1)
        inputs = torch.from_numpy(np.ascontiguousarray(img_chw)).float()

        # ---- data sample ----
        data_sample = SegDataSample()

        # ✅ metainfo (mmseg inference에 필요)
        h, w = img.shape[:2]
        # ori_shape: 원본 이미지 크기. LoadImageFromFile 단계에서 있으면 그걸 사용, 없으면 현재 img로 채움
        ori_shape = results.get("ori_shape", (h, w))
        data_sample.set_metainfo(dict(
            img_path=results.get("img_path", results.get("filename", None)),
            ori_shape=ori_shape,         # ✅ KeyError 해결 포인트
            img_shape=(h, w),
            pad_shape=(h, w),
            scale_factor=results.get("scale_factor", 1.0),
            flip=results.get("flip", False),
            flip_direction=results.get("flip_direction", None),
        ))

        # ---- gt ----
        gt = results.get("gt_multi_hot", None)
        if gt is not None:
            gt = np.ascontiguousarray(gt)  # negative stride 방지
            gt_tensor = torch.from_numpy(gt).float()  # (C,H,W)
            # 멀티라벨이지만 mmseg는 gt_sem_seg라는 키로 들고가도 됨(우린 커스텀 loss/metric에서 사용)
            data_sample.gt_sem_seg = PixelData(data=gt_tensor)

        results["inputs"] = inputs
        results["data_samples"] = data_sample
        return results
