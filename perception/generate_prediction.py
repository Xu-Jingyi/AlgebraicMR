import os, glob, ast, cv2, mmcv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import xml.etree.ElementTree as ET
import warnings

from random import randint
from PIL import Image
from mmdet.apis import inference_detector, init_detector, show_result_pyplot
from mmdet.core.evaluation.mean_ap import eval_map, tpfp_default
from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps
from utils import *


def infer_atb_with_types_bbox(types_bbox, atb_result, thr):
    atb_scores = []
    types_bbox = np.expand_dims(types_bbox, axis=0)
    for b, j in enumerate(atb_result):
        if bbox_overlaps(types_bbox, j).size == 0:
            # if no bbox set zeros_bbox and score to 0
            atb_scores.append(np.array([0], dtype=np.float32))
        else:
            # if max iou is above a certain threshold
            if np.max(bbox_overlaps(types_bbox, j)) > thr:
                # take index of color bbox, store bbox and score
                idx = np.argmax(bbox_overlaps(types_bbox, j))
                atb_scores.append(atb_result[b][idx, -1:])
            # if below the threshold set zeros_bbox and score to 0
            else:
                atb_scores.append(np.array([0], dtype=np.float32))
    return np.argmax(np.stack(atb_scores))


TYPE_VALUES = ["triangle", "square", "pentagon", "hexagon", "circle"]
SIZE_VALUES = ['0.4', '0.5', '0.6', '0.7', '0.8', '0.9']
COLOR_VALUES = ['255', '224', '196', '168', '140', '112', '84', '56', '28', '0']
BBOX_VALUES = [(0.5, 0.25, 0.5, 0.5), (0.5, 0.75, 0.5, 0.5), (0.25, 0.5, 0.5, 0.5), (0.75, 0.5, 0.5, 0.5),
               (0.5, 0.5, 1, 1), (0.42, 0.42, 0.15, 0.15), (0.83, 0.83, 0.33, 0.33), (0.16, 0.5, 0.33, 0.33),
               (0.16, 0.83, 0.33, 0.33), (0.5, 0.16, 0.33, 0.33), (0.83, 0.16, 0.33, 0.33), (0.83, 0.5, 0.33, 0.33),
               (0.42, 0.58, 0.15, 0.15), (0.16, 0.16, 0.33, 0.33), (0.58, 0.58, 0.15, 0.15), (0.58, 0.42, 0.15, 0.15),
               (0.5, 0.83, 0.33, 0.33), (0.5, 0.5, 0.33, 0.33), (0.75, 0.25, 0.5, 0.5), (0.25, 0.75, 0.5, 0.5),
               (0.25, 0.25, 0.5, 0.5), (0.75, 0.75, 0.5, 0.5)]
ABS_SIZE_VALUES = ['(0.6, 0.15)', '(0.7, 0.15)', '(0.8, 0.15)', '(0.9, 0.15)',
               '(0.4, 0.33)', '(0.5, 0.33)', '(0.6, 0.33)', '(0.7, 0.33)', '(0.8, 0.33)', '(0.9, 0.33)',
               '(0.4, 0.5)', '(0.5, 0.5)', '(0.6, 0.5)', '(0.7, 0.5)', '(0.8, 0.5)', '(0.9, 0.5)',
               '(0.4, 1.0)', '(0.5, 1.0)', '(0.6, 1.0)', '(0.7, 1.0)', '(0.8, 1.0)', '(0.9, 1.0)']
ABS_POS_VALUES = ['(0.16, 0.16, 0.33)', '(0.16, 0.5, 0.33)', '(0.16, 0.83, 0.33)', '(0.25, 0.25, 0.5)',
           '(0.25, 0.5, 0.5)', '(0.25, 0.75, 0.5)', '(0.42, 0.42, 0.15)', '(0.42, 0.58, 0.15)',
           '(0.5, 0.16, 0.33)', '(0.5, 0.25, 0.5)', '(0.5, 0.5, 0.33)', '(0.5, 0.5, 1.0)',
           '(0.5, 0.75, 0.5)', '(0.5, 0.83, 0.33)', '(0.58, 0.42, 0.15)', '(0.58, 0.58, 0.15)',
           '(0.75, 0.25, 0.5)', '(0.75, 0.5, 0.5)', '(0.75, 0.75, 0.5)', '(0.83, 0.16, 0.33)',
           '(0.83, 0.5, 0.33)', '(0.83, 0.83, 0.33)']

POSITION_VALUES_IN_RING = ['(0.16, 0.16, 0.33)', '(0.16, 0.5, 0.33)', '(0.16, 0.83, 0.33)', '(0.5, 0.16, 0.33)', '(0.5, 0.5, 0.33)',
                   '(0.5, 0.83, 0.33)', '(0.83, 0.16, 0.33)', '(0.83, 0.5, 0.33)', '(0.83, 0.83, 0.33)',
                   '(0.25, 0.25, 0.5)', '(0.25, 0.75, 0.5)', '(0.75, 0.25, 0.5)', '(0.75, 0.75, 0.5)',
                   '(0.42, 0.42, 0.15)', '(0.42, 0.58, 0.15)', '(0.58, 0.42, 0.15)', '(0.58, 0.58, 0.15)',
                   '(0.5, 0.25, 0.5)', '(0.5, 0.5, 1.0)', '(0.5, 0.75, 0.5)', '(0.25, 0.5, 0.5)', '(0.75, 0.5, 0.5)', 'dummy']
NUM_VALUES_IN_RING = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
TYPE_VALUES_IN_RING = ["triangle", "square", "pentagon", "hexagon", "circle"]
SIZE_VALUES_IN_RING = ['(0.6, 0.15)', '(0.7, 0.15)', '(0.8, 0.15)', '(0.9, 0.15)',
               '(0.4, 0.33)', '(0.5, 0.33)', '(0.6, 0.33)', '(0.7, 0.33)', '(0.8, 0.33)', '(0.9, 0.33)',
               '(0.4, 0.5)', '(0.5, 0.5)', '(0.6, 0.5)', '(0.7, 0.5)', '(0.8, 0.5)', '(0.9, 0.5)',
               '(0.4, 1.0)', '(0.5, 1.0)', '(0.6, 1.0)', '(0.7, 1.0)', '(0.8, 1.0)', '(0.9, 1.0)']
COLOR_VALUES_IN_RING = ['255', '224', '196', '168', '140', '112', '84', '56', '28', '0']

# positions ignore first
consts_names = POSITION_VALUES_IN_RING + NUM_VALUES_IN_RING + TYPE_VALUES_IN_RING + SIZE_VALUES_IN_RING + COLOR_VALUES_IN_RING

iou_thr = 0.9
data_dir = '../data/RAVEN-10000'
midformat_data_dir = '../data/midformat_Raven/rpm-600-2000-2000/test/'
fig_configs = os.listdir(data_dir)
prediction_f = open('prediction_RAVEN_iou_test_%.1f_wo_com.txt'%iou_thr, 'a+')
# ground_truth_f = open('ground_truth.txt', 'a+')
# acc = 0
# total_instance_number = 0

device = 'cuda:0'
# types_model_dir = './work_dirs/perception-types--D12-08-2022--T11-38-24--0.03/'
types_model_dir = './work_dirs/perception-types--D13-09-2022--T11-14-11--0.003'
config = glob.glob(os.path.join(types_model_dir, '*.py'))[0]
checkpoint = os.path.join(types_model_dir, 'epoch_12.pth')
type_model = init_detector(config, checkpoint, device=device)

# colors_model_dir = './work_dirs/perception-colors--D13-08-2022--T07-48-03--0.03/'
colors_model_dir = './work_dirs/perception-colors--D11-09-2022--T19-09-26--0.003'
config = glob.glob(os.path.join(colors_model_dir, '*.py'))[0]
checkpoint = os.path.join(colors_model_dir, 'epoch_12.pth')
color_model = init_detector(config, checkpoint, device=device)

# sizes_model_dir = './work_dirs/perception-abssizes--D12-08-2022--T11-39-07--0.03/'
sizes_model_dir = './work_dirs/perception-abssizes--D07-09-2022--T04-19-21--0.003'
config = glob.glob(os.path.join(sizes_model_dir, '*.py'))[0]
checkpoint = os.path.join(sizes_model_dir, 'epoch_12.pth')
size_model = init_detector(config, checkpoint, device=device)

# positions_model_dir = './work_dirs/perception-absposition--D13-08-2022--T08-47-37--0.003/'
positions_model_dir = './work_dirs/perception-absposition--D11-09-2022--T19-09-07--0.003'
config = glob.glob(os.path.join(positions_model_dir, '*.py'))[0]
checkpoint = os.path.join(positions_model_dir, 'epoch_12.pth')
position_model = init_detector(config, checkpoint, device=device)

for fig_config in fig_configs:
    # get the fig config directory
    config_dir = os.path.join(data_dir, fig_config)
    suffix = 'test.npz'

    # collect all the suffix specified files in the fig config directory
    fig_config_rpm_list = [os.path.join(config_dir, i) for i in os.listdir(config_dir) if suffix in i]
    fig_config_rpm_list.sort()

    for npzfilename in fig_config_rpm_list:
        target = np.load(npzfilename)["target"]
        xmlfilename = npzfilename[:-3] + 'xml'
        img_id = genid(xmlfilename)
        
        # writing annotations for the 16 panels
        for i in range(16):
            ann = []
            imgfilename = midformat_data_dir + img_id + str(i).rjust(2, '0') + ".jpg"

            
            type_result = inference_detector(type_model, imgfilename)
            color_result = inference_detector(color_model, imgfilename)
            size_result = inference_detector(size_model, imgfilename)
            position_result = inference_detector(position_model, imgfilename)
            # enumerate over all the type detections
            ann_pred = []
            number_pred = np.sum([len(x) for x in type_result])
            for cls, ct in enumerate(type_result):
                if len(ct) == 0:
                    # empty when no bboxes is detected for type cls
                    continue
                else:
                    # iterate over all the bboxes of type cls when nonempty
                    for k in range(len(ct)):
                        # infer color for each bbox detected for entity of type i
                        inferred_color = infer_atb_with_types_bbox(type_result[cls][k], color_result, iou_thr)
                        inferred_size = infer_atb_with_types_bbox(type_result[cls][k], size_result, iou_thr)
                        inferred_position = infer_atb_with_types_bbox(type_result[cls][k], position_result, iou_thr)
                        position = consts_names.index(ABS_POS_VALUES[inferred_position])
                        number = consts_names.index(str(number_pred))
                        type = consts_names.index(TYPE_VALUES[cls])
                        size = consts_names.index(ABS_SIZE_VALUES[inferred_size])
                        color = consts_names.index(COLOR_VALUES[inferred_color])
                        ann_pred.append([position, number, type, size, color])

            
            prediction_f.write('%s %s %d %d %s \n'%(fig_config, os.path.basename(xmlfilename)[:-4], i, target, str(ann_pred)))
            prediction_f.flush()






