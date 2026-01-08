# mmseg/evaluation/metrics/multilabel_dice_metric.py
import numpy as np
import torch
import torch.nn.functional as F
from mmengine.evaluator import BaseMetric
from mmseg.registry import METRICS

# ✅ 멀티라벨 클래스 순서(29개, background 없음) - 학습/마스크 생성과 반드시 동일해야 함
CLASSES = [
    'finger-1','finger-2','finger-3','finger-4','finger-5',
    'finger-6','finger-7','finger-8','finger-9','finger-10',
    'finger-11','finger-12','finger-13','finger-14','finger-15',
    'finger-16','finger-17','finger-18','finger-19',
    'Trapezium','Trapezoid','Capitate','Hamate','Scaphoid','Lunate',
    'Triquetrum','Pisiform','Radius','Ulna'
]

@METRICS.register_module()
class MultiLabelDiceMetric(BaseMetric):
    """
    Multi-label Dice for segmentation.
    - pred: data_sample["seg_logits"] or data_sample.seg_logits  (logits, CxHxW)
    - gt  : data_sample["gt_sem_seg"] or data_sample["gt_multi_hot"] (CxHxW)
    """

    def __init__(self, thresholds=0.5, collect_device='cpu', prefix=None,
                 print_table=False, **kwargs):
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.thresholds = thresholds
        self.print_table = print_table  # ✅ 콘솔에 표로 출력하고 싶으면 True

    def _get(self, s, key: str):
        if isinstance(s, dict):
            return s.get(key, None)
        return getattr(s, key, None)

    def _unwrap_to_tensor(self, obj):
        """PixelData / Tensor / dict wrapping 모두 tensor로 변환."""
        if obj is None:
            return None

        if hasattr(obj, "data"):  # PixelData
            obj = obj.data

        while isinstance(obj, dict):
            if "data" in obj:
                obj = obj["data"]
                continue
            if "seg_logits" in obj:
                obj = obj["seg_logits"]
                continue
            if "gt_sem_seg" in obj:
                obj = obj["gt_sem_seg"]
                continue
            break

        if hasattr(obj, "data"):
            obj = obj.data

        if torch.is_tensor(obj):
            return obj
        if isinstance(obj, np.ndarray):
            return torch.from_numpy(obj)
        return None

    def process(self, data_batch, data_samples):
        for s in data_samples:
            # logits
            logits_obj = self._get(s, "seg_logits") or self._get(s, "pred_logits")
            logits = self._unwrap_to_tensor(logits_obj)
            if logits is None:
                raise KeyError(
                    f"Cannot find tensor seg_logits in data_sample keys="
                    f"{list(s.keys()) if isinstance(s, dict) else 'SegDataSample'}"
                )

            if logits.ndim == 4:
                logits = logits.squeeze(0)  # (C,H,W)
            prob = torch.sigmoid(logits)

            # gt
            gt_obj = self._get(s, "gt_sem_seg") or self._get(s, "gt_multi_hot")
            gt = self._unwrap_to_tensor(gt_obj)
            if gt is None:
                raise KeyError("Cannot find tensor gt_sem_seg/gt_multi_hot in data_sample.")

            if gt.ndim == 4:
                gt = gt.squeeze(0)  # (C,H,W)

            # ✅ pred가 ori_shape로 올라가면 GT 크기로 맞춰서 계산
            if prob.shape[-2:] != gt.shape[-2:]:
                prob = F.interpolate(
                    prob.unsqueeze(0),
                    size=gt.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)

            C = prob.shape[0]

            # threshold tensor (C,1,1)
            if isinstance(self.thresholds, (float, int)):
                thr = torch.full((C, 1, 1), float(self.thresholds), device=prob.device)
            else:
                thr_list = list(self.thresholds)
                assert len(thr_list) == C
                thr = torch.tensor(thr_list, device=prob.device).view(C, 1, 1)

            pred_bin = (prob > thr).float()

            eps = 1e-7
            dices = []
            for c in range(C):
                p = pred_bin[c].reshape(-1)
                g = gt[c].float().reshape(-1)
                tp = (p * g).sum()
                fp = (p * (1 - g)).sum()
                fn = ((1 - p) * g).sum()
                dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
                dices.append(dice.detach().cpu().item())

            self.results.append(dict(dice=dices))

    def compute_metrics(self, results):
        all_dice = np.array([r["dice"] for r in results], dtype=np.float32)  # (N,C)
        mean_per_class = all_dice.mean(axis=0)  # (C,)
        mDice = float(mean_per_class.mean())

        metrics = {"mDice": mDice}

        # ✅ 클래스별 Dice를 metric key로 추가 (logger / wandb에서 확인 가능)
        # 키가 길어지는 게 싫으면 "Dice/{i:02d}_{name}" 처럼 바꿔도 됨
        for i, name in enumerate(CLASSES[:len(mean_per_class)]):
            metrics[f"Dice/{name}"] = float(mean_per_class[i])

        # ✅ 예전처럼 표로 보고 싶으면 옵션으로 출력
        if self.print_table:
            # 간단한 표 형태(외부 패키지 설치 없이)
            lines = []
            lines.append("+----------------+--------+")
            lines.append("| Class          | Dice   |")
            lines.append("+----------------+--------+")
            for i, name in enumerate(CLASSES[:len(mean_per_class)]):
                lines.append(f"| {name:<14} | {mean_per_class[i]*100:6.2f} |")
            lines.append("+----------------+--------+")
            print("\n".join(lines))

        return metrics
