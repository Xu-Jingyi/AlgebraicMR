_base_ = '../libra_retinanet_r50_fpn_1x_coco.py'

dataset_type = 'IRAVENAbsposition'

classes = ('(0.16, 0.16, 0.33)', '(0.16, 0.5, 0.33)', '(0.16, 0.83, 0.33)', '(0.5, 0.16, 0.33)', 
           '(0.5, 0.5, 0.33)', '(0.5, 0.83, 0.33)', '(0.83, 0.16, 0.33)', '(0.83, 0.5, 0.33)', 
           '(0.83, 0.83, 0.33)', '(0.25, 0.25, 0.5)', '(0.25, 0.75, 0.5)', '(0.75, 0.25, 0.5)', 
           '(0.75, 0.75, 0.5)', '(0.42, 0.42, 0.15)', '(0.42, 0.58, 0.15)', '(0.58, 0.42, 0.15)', 
           '(0.58, 0.58, 0.15)', '(0.5, 0.25, 0.5)', '(0.5, 0.5, 1.0)', '(0.5, 0.75, 0.5)', 
           '(0.25, 0.5, 0.5)', '(0.75, 0.5, 0.5)')

_base_ = '../libra_retinanet_r50_fpn_1x_coco.py'

data = dict(
    samples_per_gpu=16,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        # explicitly add your class names to the field `classes`
        classes=classes,
        ann_file='train-abspositions.txt',
        img_prefix='',
        data_root = '../data/midformat/rpm-600-2000-2000/train'),
    val=dict(
        type=dataset_type,
        # explicitly add your class names to the field `classes`
        classes=classes,
        ann_file='val-abspositions.txt',
        img_prefix='',
        data_root = '../data/midformat/rpm-600-2000-2000/val'),
    test=dict(
        type=dataset_type,
        # explicitly add your class names to the field `classes`
        classes=classes,
        ann_file='test-abspositions.txt',
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
load_from = "configs/libra_retinanet_r50_fpn_1x_coco_20200205-804d94ce.pth"
