# configs/_local/upernet_swin_base_bone_ml.py

# ✅ 1) Swin-B + UPerNet 베이스로 교체
_base_ = '../swin/swin-base-patch4-window12-in22k-384x384-pre_upernet_8xb2-160k_ade20k-512x512.py'

custom_imports = dict(
    imports=[
        'mmseg.datasets.bone_dataset_ml',
        'mmseg.datasets.transforms.multilabel_transforms',
        'mmseg.datasets.transforms.pack_multilabel',
        'mmseg.evaluation.metrics.multilabel_dice_metric',
    ],
    allow_failed_imports=False
)

data_root = '/data/ephemeral/home/data'
dataset_type = 'BoneDatasetML'

# ✅ 멀티라벨: background 없음(29)
num_classes = 29

norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)

model = dict(
    decode_head=dict(
        num_classes=num_classes,
        out_channels=num_classes,
        norm_cfg=norm_cfg,
        loss_decode=[
            dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),  # BCEWithLogits
            dict(type='DiceLoss', use_sigmoid=True, loss_weight=1.0),
        ],
    ),
    # auxiliary_head=dict(
    #     num_classes=num_classes,
    #     out_channels=num_classes,
    #     norm_cfg=norm_cfg,
    #     loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=0.4),
    # ),
    auxiliary_head=None,  # ✅ aux head OFF (메모리 크게 절약)
    backbone=dict(with_cp=True),  # ✅ checkpointing
)

# -------------------------
# ✅ 2) 2048 원본 크기 그대로 학습/검증 파이프라인
# -------------------------
# 너의 커스텀 멀티라벨 변환(RandomResizeML/RandomCropML/RandomFlipML) 그대로 사용 :contentReference[oaicite:1]{index=1}
# PackSegInputsML도 그대로 사용 :contentReference[oaicite:2]{index=2}
train_pipeline = [
    dict(type='LoadImageFromFile'),

    # (권장) 2048 고정에 가깝게: ratio_range를 좁혀서 "거의 원본 유지"
    dict(type='RandomResizeML', scale=(2048, 2048), ratio_range=(0.9, 1.1), keep_ratio=True),

    # ✅ 핵심: crop도 2048로 (= 사실상 full image에 가깝게)
    # dict(type='RandomCropML', crop_size=(2048, 2048)), #2048 그대로”를 원하면, 사실 RandomCropML(2048,2048)은 빼는 게 더 정확

    dict(type='RandomFlipML', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputsML'),
]

val_pipeline = [
    dict(type='LoadImageFromFile'),

    # val은 완전 고정
    dict(type='RandomResizeML', scale=(2048, 2048), ratio_range=(1.0, 1.0), keep_ratio=True),

    dict(type='PackSegInputsML'),
]

train_dataloader = dict(
    batch_size=1,
    num_workers=2, #안정적이게 2로 변경 (workers 안정화)
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='train/DCM', seg_map_path='masks_train_ml'),
        ann_file='train_list_ml.txt',
        pipeline=train_pipeline,
    ),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='train/DCM', seg_map_path='masks_train_ml'),
        ann_file='val_list_ml.txt',
        pipeline=val_pipeline,
    ),
)

test_dataloader = val_dataloader

# -------------------------
# ✅ 3) Train schedule (너 기존과 동일)
# -------------------------
train_cfg = dict(type='IterBasedTrainLoop', max_iters=40000, val_interval=500)

param_scheduler = [
    dict(type='LinearLR', begin=0, end=1500, by_epoch=False, start_factor=1e-6),
    dict(type='PolyLR', begin=1500, end=40000, by_epoch=False, eta_min=0.0, power=1.0),
]

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=0,
        save_best='mDice',
        rule='greater',
        max_keep_ckpts=1,
        save_last=False,
    ),
    logger=dict(type='LoggerHook', interval=1000, log_metric_by_epoch=False),
)

# -------------------------
# ✅ 4) 멀티라벨 mDice 평가 (그대로)
# -------------------------
val_evaluator = dict(_delete_=True, type='MultiLabelDiceMetric', thresholds=0.5)
test_evaluator = dict(_delete_=True, type='MultiLabelDiceMetric', thresholds=0.5)

# -------------------------
# ✅ 5) AMP(fp16) 적용 (가장 중요)
# -------------------------
# 베이스 config의 optim_wrapper를 덮어쓰기 위해 _delete_ 사용
optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',
    loss_scale='dynamic',
    optimizer=dict(type='AdamW', lr=6e-5, betas=(0.9, 0.999), weight_decay=0.01),
    clip_grad=dict(max_norm=1.0, norm_type=2),
)

# wandb는 원하면 프로젝트/이름만 바꿔서 유지
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='WandbVisBackend',
        init_kwargs=dict(
            project='bone-seg-swin-L',
            name='upernet_swin_b_2048_40k_ml',
        ),
        log_code_name=None,
    ),
]
visualizer = dict(type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')
