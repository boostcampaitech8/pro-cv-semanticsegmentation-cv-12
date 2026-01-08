# mmseg/datasets/bone_dataset_ml.py
from pathlib import Path
import numpy as np

from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset

CLASSES = [
    'finger-1','finger-2','finger-3','finger-4','finger-5',
    'finger-6','finger-7','finger-8','finger-9','finger-10',
    'finger-11','finger-12','finger-13','finger-14','finger-15',
    'finger-16','finger-17','finger-18','finger-19',
    'Trapezium','Trapezoid','Capitate','Hamate','Scaphoid','Lunate',
    'Triquetrum','Pisiform','Radius','Ulna'
]

@DATASETS.register_module()
class BoneDatasetML(BaseSegDataset):
    METAINFO = dict(classes=tuple(CLASSES), palette=[(0, 0, 0)] * len(CLASSES))

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load_data_list(self):
        ann_path = Path(self.data_root) / self.ann_file
        names = [
            line.strip()
            for line in ann_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        data_list = []
        for name in names:
            rel = Path(name)  # e.g. ID338/image1664....png

            img_path = Path(self.data_root) / self.data_prefix["img_path"] / rel

            # ✅ 핵심: rel.parent 포함해서 마스크 경로 구성
            # masks_train_ml/ID338/image1664....npz
            mask_path = (
                Path(self.data_root)
                / self.data_prefix["seg_map_path"]
                / rel.parent
                / f"{rel.stem}.npz"
            )

            data_list.append(dict(
                img_path=str(img_path),
                seg_map_path=str(mask_path),
                seg_fields=[],
            ))
        return data_list

    def get_data_info(self, idx):
        info = super().get_data_info(idx)

        mask_path = Path(info["seg_map_path"])
        if not mask_path.exists():
            raise FileNotFoundError(
                f"mask not found: {mask_path}\n"
                f"Expected structure: <data_root>/{self.data_prefix['seg_map_path']}/<IDxxx>/<stem>.npz\n"
                f"Example: masks_train_ml/ID338/image1664848616528.npz"
            )

        with np.load(str(mask_path)) as z:
            gt = z["mask"]  # (29,H,W) uint8

        if gt.ndim != 3:
            raise ValueError(f"gt mask must be (C,H,W), got {gt.shape} at {mask_path}")
        if gt.shape[0] != len(CLASSES):
            raise ValueError(f"gt first dim must be {len(CLASSES)}, got {gt.shape[0]} at {mask_path}")

        info["gt_multi_hot"] = gt  # ✅ uint8 유지 (메모리 안정)
        return info
