# The new config inherits a base config to highlight the necessary modification
_base_ = '../faster_rcnn/faster_rcnn_r101_fpn_1x_coco.py'

# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=1),))

# Modify dataset related settings
# dataset_type='COCODataset'
work_dir='faster_rcnn_r101_sim10k_detection_dg_pd'
classes = ("car",)
auto_scale_lr = dict(enable=True, base_batch_size=32)

optimizer = dict(type="SGD", lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)

# learning policy
lr_config = dict(
    policy="step",
    warmup="linear",
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[6000, 8000],
)

runner = dict(type="IterBasedRunner", max_iters=10000)

evaluation = dict(interval=1000, metric="bbox", save_best="bbox_mAP_50")

log_config = dict(
    interval=100,
    hooks=[
        dict(type="TextLoggerHook"),
    ],
)

checkpoint_config = dict(interval=1000)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(
        type='Resize', img_scale=[(2048, 800), (2048, 1024)], keep_ratio=True),
    dict(type="PhotoMetricDistortion"),
    dict(type="RandomFlip", flip_ratio=0.5),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size_divisor=32),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels"]),
]
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(2048, 1024),
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="Pad", size_divisor=32),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]

data = dict(
    train=dict(
        type="CocoDataset",
        img_prefix="/srv/share4/datasets/sim10k/10k/Sim10k/JPEGImages/",
        classes=classes,
        ann_file="/srv/share4/datasets/sim10k/10k/Sim10k/COCOAnnotations/voc2012_annotations.json",
        pipeline=train_pipeline,
    ),
    val=dict(
        type="CityscapesDataset",
        img_prefix="/srv/datasets/cityscapes_DG/leftImg8bit/val/",
        classes=classes,
        ann_file="/srv/datasets/cityscapes_DG/annotations/instancesonly_filtered_gtFine_val.json",
        pipeline=test_pipeline,
    ),
    test=dict(
        type="CityscapesDataset",
        img_prefix="/srv/datasets/cityscapes_DG/leftImg8bit/val/",
        classes=classes,
        ann_file="/srv/datasets/cityscapes_DG/annotations/instancesonly_filtered_gtFine_val.json",
        pipeline=test_pipeline,
    ),
)
