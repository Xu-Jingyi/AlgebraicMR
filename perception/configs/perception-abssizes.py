_base_ = '../libra_retinanet_r50_fpn_1x_coco.py'

dataset_type = 'IRAVENAbssize'
classes = ((0.6, 0.15), (0.7, 0.15), (0.8, 0.15), (0.9, 0.15),
               (0.4, 0.33), (0.5, 0.33), (0.6, 0.33), (0.7, 0.33), (0.8, 0.33), (0.9, 0.33),
               (0.4, 0.5), (0.5, 0.5), (0.6, 0.5), (0.7, 0.5), (0.8, 0.5), (0.9, 0.5),
               (0.4, 1.0), (0.5, 1.0), (0.6, 1.0), (0.7, 1.0), (0.8, 1.0), (0.9, 1.0))
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        # explicitly add your class names to the field `classes`
        classes=classes,
        ann_file='train-abssizes.txt',
        img_prefix='',
        data_root = '../data/midformat/rpm-600-2000-2000/train'),
    val=dict(
        type=dataset_type,
        # explicitly add your class names to the field `classes`
        classes=classes,
        ann_file='val-abssizes.txt',
        img_prefix='',
        data_root = '../data/midformat/rpm-600-2000-2000/val'),
    test=dict(
        type=dataset_type,
        # explicitly add your class names to the field `classes`
        classes=classes,
        ann_file='test-abssizes.txt',
        img_prefix='',
        data_root = '../data/midformat/rpm-600-2000-2000/test'))

model = dict(
    bbox_head=dict(type='RetinaHead',
                   num_classes=22),
    train_cfg=dict(assigner=dict(min_pos_iou=0.5)),
    test_cfg=dict(score_thr=0.95),
            )
evaluation = dict(interval=12, metric='mAP')
data_root = '../data/midformat/rpm-600-2000-2000'
load_from = "./configs/libra_retinanet_r50_fpn_1x_coco_20200205-804d94ce.pth"
