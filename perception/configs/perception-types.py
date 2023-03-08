_base_ = '../libra_retinanet_r50_fpn_1x_coco.py'

dataset_type = 'IRAVENType'
classes = ('triangle', 'square', 'pentagon', 'hexagon', 'circle')

data = dict(
    samples_per_gpu=16,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        # explicitly add your class names to the field `classes`
        classes=classes,
        ann_file='train-types.txt',
        img_prefix='',
        data_root = '../data/midformat/rpm-600-2000-2000/train'),
    val=dict(
        type=dataset_type,
        # explicitly add your class names to the field `classes`
        classes=classes,
        ann_file='val-types.txt',
        img_prefix='',
        data_root = '../data/midformat/rpm-600-2000-2000/val'),
    test=dict(
        type=dataset_type,
        # explicitly add your class names to the field `classes`
        classes=classes,
        ann_file='test-types.txt',
        img_prefix='',
        data_root = '../data/midformat/rpm-600-2000-2000/test'))

model = dict(
    bbox_head=dict(num_classes=5),
    train_cfg=dict(assigner=dict(min_pos_iou=0.5)),
    test_cfg=dict(score_thr=0.99),
            )
evaluation = dict(interval=12, metric='mAP')
data_root = '../data/midformat/rpm-600-2000-2000'
load_from = "configs/libra_retinanet_r50_fpn_1x_coco_20200205-804d94ce.pth"
