import os, glob, ast, cv2, mmcv, random

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import xml.etree.ElementTree as ET

# from random import randint
from tqdm import trange
from PIL import Image
from mmdet.core.evaluation.mean_ap import eval_map, tpfp_default
from mmdet.apis import inference_detector, init_detector, show_result_pyplot
from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps

# constants
SIZE_VALUES = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
TYPE_VALUES = ["triangle", "square", "pentagon", "hexagon", "circle"]
COLOR_VALUES = [255, 224, 196, 168, 140, 112, 84, 56, 28, 0]
ABS_SIZE_VALUES = [(0.6, 0.15), (0.7, 0.15), (0.8, 0.15), (0.9, 0.15),
               (0.4, 0.33), (0.5, 0.33), (0.6, 0.33), (0.7, 0.33), (0.8, 0.33), (0.9, 0.33),
               (0.4, 0.5), (0.5, 0.5), (0.6, 0.5), (0.7, 0.5), (0.8, 0.5), (0.9, 0.5),
               (0.4, 1.0), (0.5, 1.0), (0.6, 1.0), (0.7, 1.0), (0.8, 1.0), (0.9, 1.0)]
ABS_POS_VALUES = [(0.16, 0.16, 0.33), (0.16, 0.5, 0.33), (0.16, 0.83, 0.33), (0.5, 0.16, 0.33), (0.5, 0.5, 0.33),
                   (0.5, 0.83, 0.33), (0.83, 0.16, 0.33), (0.83, 0.5, 0.33), (0.83, 0.83, 0.33),
                   (0.25, 0.25, 0.5), (0.25, 0.75, 0.5), (0.75, 0.25, 0.5), (0.75, 0.75, 0.5),
                   (0.42, 0.42, 0.15), (0.42, 0.58, 0.15), (0.58, 0.42, 0.15), (0.58, 0.58, 0.15),
                   (0.5, 0.25, 0.5), (0.5, 0.5, 1.0), (0.5, 0.75, 0.5), (0.25, 0.5, 0.5), (0.75, 0.5, 0.5)]

FIG_CONFIGS = ['center_single', 'in_center_single_out_center_single', 'distribute_four',
               'left_center_single_right_center_single', 'up_center_single_down_center_single',
               'in_distribute_four_out_center_single', 'distribute_nine']
IMAGE_SIZE = 160


# util functions

def genid(filename):
    """
    Generated a 8 digit identification number prefixed with `I-RAVEN-` with the following name style:
    
    1: {0 : 'center_single', 
        1 : 'in_center_single_out_center_single', 
        2 : 'distribute_four', 
        3 : 'left_center_single_right_center_single', 
        4 : 'up_center_single_down_center_single', 
        5 : 'in_distribute_four_out_center_single', 
        6 : 'distribute_nine' }
    2-5: sample number within the configuration, ranges from 0000 to 9999
    6: {0 : 'train', 1 : 'val', 2 : 'test'}
    7-8: panel number which ranges from 00 to 15
     
    Args:
        filename (str): path to a I-RAVEN dataset filename with either .npz or .xml extension.
    """
    fn = os.path.basename(filename)
    fn = fn.replace('_', ' ')
    fd = fn.split()
    num = int(fd[1])
    subset = fd[2][:-4]

    subfilename = filename.replace('/' + os.path.basename(filename), '')

    a = str(FIG_CONFIGS.index(os.path.basename(subfilename)))
    b = str(num).rjust(4, '0')

    if subset == 'train':
        c = str(0)
    elif subset == 'val':
        c = str(1)
    else:
        c = str(2)

    return 'IRAVEN-' + a + b + c


def to_xml(filename):
    """
    Function which replaces any three character extension to .xml.
    Used for converting filenames with .npz extension to .xml.
        
    Args:
        filename (str): a file name with extension with three characters e.g. .npz
    """
    return filename[:-3] + 'xml'


def parse_xml(xmlf):
    """
    Parses the given xml file name and returns lists of lists real_bboxes, types, colors, sizes and positions
    where each list in the list corresponds to a panel.
    
    Args:
        xmlf: a file name with .xml extension 
    """

    tree = ET.parse(xmlf)
    root = tree.getroot()
    root_len = len(root[0])

    real_bboxes = []
    typs = []
    colors = []
    sizes = []
    positions = []

    for panel in range(len(root[0])):
        panel_real_bboxes = []
        panel_typs = []
        panel_colors = []
        panel_sizes = []
        panel_positions = []

        for component in root[0][panel][0]:
            for entity in component[0]:
                panel_real_bboxes.append(entity.attrib["real_bbox"])
                panel_positions.append(entity.attrib["bbox"])
                panel_typs.append(int(entity.attrib["Type"])-1)
                panel_colors.append(int(entity.attrib["Color"]))
                panel_sizes.append(int(entity.attrib["Size"]))

        real_bboxes.append(panel_real_bboxes)
        typs.append(panel_typs)
        colors.append(panel_colors)
        sizes.append(panel_sizes)
        positions.append(panel_positions)
    return real_bboxes, typs, colors, sizes, positions


def to_bbox(real_bbox, image_size=IMAGE_SIZE, intg=True, df='xy'):
    """
    Converts the string real_bbox values extracted from xml labels with format:
    
            [center_y, center_x, width, height]
    
    where 0<= center_x, center_y, width, height <= 1, to `wh` format:
        
            [top_x, top_y, width, height]
    
    where top_x, top_y, width, height are pixel values within IMAGE_SIZE = 160, or to `xy` format
    
            [top_x, top_y, bot_x, bot_y]
    
    Args:
        real_bbox (str): string output of bbox from XML file
        image_size (int): the size of the image where the bbox is defined on
        intg (boolean): indicates choice of integer bbox values. Integer bbox values are used for plotting bboxes.
        df (str): bbox dataformat, `xy` indicates [top_x, top_y, bot_x, bot_y] and `wh` indicates [top_x, top_y, width, height]
    
    Note: for middleformat, intg = False, df = 'xy'
    """
    cy, cx, w, h = ast.literal_eval(real_bbox)

    # max and min to ensure that the bbox values remain within IMAGE_SIZE 

    if df == 'xy':
        a = max((cx - w / 2) * image_size, 0)
        b = max((cy - h / 2) * image_size, 0)
        c = min((cx + w / 2) * image_size, image_size)
        d = min((cy + h / 2) * image_size, image_size)

    elif df == 'wh':

        a = max((cx - w / 2) * image_size, 0)
        b = max((cy - h / 2) * image_size, 0)
        c = w * image_size
        d = h * image_size

    else:
        raise Exception('Wrong format provided. Formats allowed are `xy` or `wh`.')

    if intg:

        return np.ceil(np.array([a, b, c, d])).astype('int')
    else:
        return np.array([a, b, c, d])


def gen_abssize_labels(sizes, positions):
    """
    Generate the absolute size of an entity using the size and position labels from the xml files.
    
    Args:
        sizes (list): sizes extracted from an xml label using `parse_xml`
        positions (list): positions extracted from an xml label using `parse_xml`
    """

    abs_size_labels = []

    for idx, sp in enumerate(zip(sizes, positions)):
        assert len(sp[0]) == len(sp[1])
        panel_labels = []
        for i, j in zip(sp[0], sp[1]):
            cls = tuple((SIZE_VALUES[i], ast.literal_eval(j)[-1]))
            panel_labels.append(ABS_SIZE_VALUES.index(cls))
        abs_size_labels.append(panel_labels)

    assert len(abs_size_labels) == len(sizes)
    assert len([j for i in abs_size_labels for j in i]) == len([j for i in sizes for j in i])

    return abs_size_labels


def gen_absposition_labels(positions):
    """
    Generate the absolute position labels of an entity using ABS_POS_VALUES derived from BBOX_VALUES.
    
    Args:
        sizes (list): sizes extracted from an xml label using `parse_xml`
        positions (list): positions extracted from an xml label using `parse_xml`
    """

    pos_labels = []

    for p in positions:
        panel_pos = []
        for e in p:
            pos = tuple(ast.literal_eval(e)[:3])
            idx = ABS_POS_VALUES.index(pos)
            panel_pos.append(idx)
        pos_labels.append(panel_pos)

    return pos_labels


def annotate_label(filename, real_bboxes, attribute_label, save_dir, ann_name, df='xy'):
    """
    Generates annotation labels for a single attribute which is specified by `label`
    
    Args:
        filename (str): filename of the 
        real_bboxes: the real bounding boxes of the entities obtained from `parse_xml`.
        attribute_label: the labels of the attribute obtained from `parse_xml`.
        save_dir (str): directory where the generated jpeg image files are saved.
        ann_name (str): filename to save the annotation labels.
        df (str | optional): the bbox data format to be used in the conversion from real_bbox.
    """
    img_id = genid(filename)

    if not ann_name.endswith('.txt'):
        ann_name = ann_name + '.txt'

    ann = open(os.path.join(save_dir, ann_name), "a")
    count = 0
    # writing annotations for the 16 panels
    for i in range(16):
        imgfilename = img_id + str(i).rjust(2, '0') + ".jpg"
        ann.write("#" + "\n")
        ann.write(imgfilename + "\n")
        ann.write(str(IMAGE_SIZE) + " " + str(IMAGE_SIZE) + "\n")

        l = len(real_bboxes[i])  # number of objects in panel i
        ann.write(str(l) + "\n")

        bboxes = [str(to_bbox(x, intg=False, df=df)) for x in real_bboxes[i]]

        # iterate over objects in panel i
        for j in range(l):
            d = bboxes[j][1:-1]  # remove braces of the list
            d = d.replace(",", "")  # remove commas
            t = attribute_label[i][j]  # object attrib in the bbox
            dt = d + " " + str(t)
            dt = dt.lstrip()
            ann.write(dt + "\n")

        count += 1
    assert count == 16

    ann.close()


def generate_det_ann_from_xml(xmlfilename, attribs, save_dir):
    """
    Generate detection annotation labels for a single npz based on the attribute list provided.
    
    Args:
        xmlfilename (str): filename of the xml label file to generate the annotation from.
        attribs (str or list): attribute(s) to generate the annotation for.
        save_dir (str): directory where the generated annotation files are saved.
    """

    img_id = genid(xmlfilename)
    valid_attrib_list = ['types', 'colors', 'abspositions', 'abssizes']

    if os.path.basename(xmlfilename).endswith('train.xml'):
        subset = 'train'
    elif 'val' in os.path.basename(xmlfilename):
        subset = 'val'
    elif 'test' in os.path.basename(xmlfilename):
        subset = 'test'
    else:
        raise Exception('Provided xmlfilename does not match the requirements.')

    save_dir = os.path.join(save_dir, subset)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # if single attribute string is provided, convert to a list
    if not isinstance(attribs, list):
        attrib_list = [attribs]
    else:
        attrib_list = attribs

    # check that provided list of attributes is valid
    assert all(a in valid_attrib_list for a in attrib_list)

    real_bboxes, types, colors, sizes, positions = parse_xml(xmlfilename)

    types_label = types
    colors_label = colors
    abssizes_label = gen_abssize_labels(sizes, positions)  # using pairs of size and position labels
    abspositions_label = gen_absposition_labels(positions)

    # generate annotation labels based on attribs list
    for attrib in attrib_list:
        if attrib == 'types':
            annotate_label(xmlfilename, real_bboxes, types_label, save_dir, subset + '-' + attrib)
        elif attrib == 'colors':
            annotate_label(xmlfilename, real_bboxes, colors_label, save_dir, subset + '-' + attrib)
        elif attrib == 'abssizes':
            annotate_label(xmlfilename, real_bboxes, abssizes_label, save_dir, subset + '-' + attrib)
        elif attrib == 'abspositions':
            annotate_label(xmlfilename, real_bboxes, abspositions_label, save_dir, subset + '-' + attrib)
        else:
            raise Exception('Invalid attribute provided. Select from `types`, `colors`, `abspositions` or `abssizes`.')


def generate_jpg_from_npz(npzfilename, save_dir):
    """
    Generate jpeg images from a single RPM npzfile.
    
    Args:
        npzfilename (str): filename of the npz data file to generate the jpeg from.
        save_dir (str): directory where the generated annotation files are saved.
    """

    if 'train' in os.path.basename(npzfilename):
        subset = 'train'
    elif 'val' in os.path.basename(npzfilename):
        subset = 'val'
    elif 'test' in os.path.basename(npzfilename):
        subset = 'test'
    else:
        raise Exception('Provided npzfilename does not match the requirements.')

    save_dir = os.path.join(save_dir, subset)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    data = np.load(npzfilename)
    image = data["image"]

    img_id = genid(npzfilename)

    assert np.shape(image)[0] == 16

    for i in range(16):  # iterate over panels in a data

        imgfilename = img_id + str(i).rjust(2, '0') + ".jpg"
        pic = Image.fromarray(image[i])
        pic.save(os.path.join(save_dir, imgfilename))


def generate_det_midformat(npzfilename, save_dir, attrib=['types', 'colors', 'abssizes', 'abspositions'], df='xy'):
    """
    Processes a npz RPM data file which contains 16 image panels into 16 jpeg images,
    with text annotation labels.
    
    Generated jpeg images follow a 8 digit name style:
    
    1: {0 : 'center_single', 
        1 : 'in_center_single_out_center_single', 
        2 : 'distribute_four', 
        3 : 'left_center_single_right_center_single', 
        4 : 'up_center_single_down_center_single', 
        5 : 'in_distribute_four_out_center_single', 
        6 : 'distribute_nine' }
    2-5: sample number within the configuration, ranges from 0000 to 9999
    6: {0 : 'train', 1 : 'val', 2 : 'test'}
    7-8: panel number which ranges from 00 to 15
    
    Args:
        filename (str): a filename path leading to a .npz data file
        save_dir (str): directory where the generated jpeg image files are saved.
        attrib (str|list): the annotation attributes label to generated.
        ann_name (str): filename to save the annotation labels.
        df (str | optional): the bbox data format to be used in the conversion from real_bbox.
    """

    xmlfilename = to_xml(npzfilename)

    generate_jpg_from_npz(npzfilename, save_dir)

    generate_det_ann_from_xml(xmlfilename, attrib, save_dir)


def process_det_midformat(data_dir, save_dir, split=(60, 20, 20), atb=['types', 'colors', 'abssizes', 'abspositions'],
                          fig_config=None, df='xy', seed=0):
    """
    Processes a list of npz RPM data files to middle format; jpeg images, with text annotation labels.
    
    
    Args:
        data_dir (str): directory where the folders contain the 7 figure configuration folders.
        save_dir (str): indicates path to the directory for saving the jpeg image files.
        attrib (str|list): the annotation attributes label to generated.
        split (tuple): 3-tuple to denote the split of train, val and test samples.
        fig_config (list): list of integer(s) of the configurations, e.g. [0,3,4]. 
                           If None, process all the configurations.
        df (str | optional): the bbox data format to be used in the conversion from real_bbox.
        seed (int): seed for the random shuffling of rpm data list.
    """
    save_dir = os.path.join(save_dir, 'rpm' + '-' + str(split[0]) + '-' + str(split[1]) + '-' + str(split[2]))

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    subsets = ['train', 'val', 'test']

    # consolidate fig configs if provided, otherwise use all fig configs
    if fig_config != None:
        fig_configs = [fig_configs[i] for i in fig_config]
    else:
        fig_configs = os.listdir(data_dir)

    # number of fig configs to be processed
    num_figconfig = len(fig_configs)

    total_rpm_list = []

    # iterate over the different subsets
    for subset, subset_size in zip(subsets, split):

        print('Collating {} {} samples from each figure configurations...'.format(subset_size, subset))

        rpm_list = []

        for n in trange(num_figconfig):

            suffix = subset + '.npz'

            # collect all the suffix specified files in the fig config directory
            fig_config_rpm_list = [i for i in os.listdir(os.path.join(data_dir, fig_configs[n])) if suffix in i]
            fig_config_rpm_list.sort()

            # get the fig config directory
            config_dir = os.path.join(data_dir, fig_configs[n])

            # get path to the RPM files
            fig_config_rpm_list = [os.path.join(config_dir, i) for i in fig_config_rpm_list]

            # shuffle with default random seed, unless provided 
            if isinstance(seed, int):
                random.seed(seed + n)
                random.shuffle(fig_config_rpm_list)

                # take the number of samples needed for the subset
                rpm_list += fig_config_rpm_list[:subset_size]

            else:
                raise Exception('Provided seed needs to be an integer.')

        num_samples = len(rpm_list)
        print(num_samples)
        assert num_samples == subset_size * 7
        total_rpm_list.append(rpm_list)

    print('Collation of samples completed.')

    if not isinstance(atb, list):
        atbs = [atb]
    else:
        atbs = atb

    total_rpm_list = [r for s in total_rpm_list for r in s]

    total_rpm_num = len(total_rpm_list)

    print('Generating samples with {} annotation labels...'.format(', '.join(atbs)))

    for idx in trange(total_rpm_num):
        generate_det_midformat(total_rpm_list[idx], save_dir, atb)
    print('Data process to middle format completed.')


def get_image_list(image_dir):
    """
    Returns the list of images in the training, validation and testing folders in the provided dataset directory.
    
    Args:
        dataset_dir: directory which contains the directory of train, val and test folders which 
                    contains the jpeg images.    
    """
    for i in os.listdir(image_dir):
        if i == 'train':
            train_list = os.listdir(os.path.join(image_dir, i))
            train_list = [i for (i, v) in zip(train_list, [i[-3:] != 'txt' for i in train_list]) if v]
        elif i == 'val':
            val_list = os.listdir(os.path.join(image_dir, i))
            val_list = [i for (i, v) in zip(val_list, [i[-3:] != 'txt' for i in val_list]) if v]
        elif i == 'test':
            test_list = os.listdir(os.path.join(image_dir, i))
            test_list = [i for (i, v) in zip(test_list, [i[-3:] != 'txt' for i in test_list]) if v]
        else:
            raise Exception('Directory does not contain `train`, `val` or `test` folders.')
    return train_list, val_list, test_list


def infer_atb(model_dir, model_checkpoint, imglist, img_data_dir, gpu_ids=0, batch_size=160):
    """
    Perform inferences of the attribute based on the provided model and trained weights(checkpoint).
    Returns results of the inferred attribute.
    
    Args:
        model_dir (str): directory to model directory.
        model_checkpoint (.pth): checkpoint to be used for the model.
        imglist (list): list of images to be inferred.
        img_data_dir (str):
        gpu_ids (int):
        batch_size (int):    
    """

    model_files = os.listdir(model_dir)
    config = glob.glob(os.path.join(model_dir, '*.py'))[0]
    checkpoint = os.path.join(model_dir, model_checkpoint)

    device = 'cuda:' + str(gpu_ids)

    nn = init_detector(config, checkpoint, device=device)

    num = len(imglist)

    atb_result = []

    if num // batch_size == num / batch_size:
        batches = num // batch_size
    else:
        batches = num // batch_size + 1

    imglist = [os.path.join(img_data_dir, i) for i in imglist]

    for b in trange(batches):
        batch_imglist = imglist[batch_size * b:batch_size * (b + 1)]
        #         batch_imglist = [os.path.join(img_data_dir,i) for i in batch_imglist]

        batch_types_result = inference_detector(nn, batch_imglist)
        atb_result += batch_types_result

    return atb_result


def infer_atb_with_types(types_bbox, atb_result, thr=0.95):
    """
    Infers the attribute of the entity within the given types bbox wrt the results 
    from perception module trained on the attribute labels.
    
    Args:
        types_bbox: a numpy array with dim (5,)
        attrib_result: a list of numpy arrays containing the bbox and 
                       scores of the detected entities wrt the attribute
        thr: a value between 0 and 1 to indicate the iou threshold 
             for the attribute bbox to be a candidate 
    """
    zeros_bbox = np.array([0, 0, 0, 0, 0], dtype=np.float32)
    types_bbox = np.expand_dims(types_bbox, axis=0)
    atb_scores = []
    cand_atb_dts = []
    for b, j in enumerate(atb_result):
        if bbox_overlaps(types_bbox, j).size == 0:

            # if no bbox set zeros_bbox and score to 0
            atb_scores.append(np.array([0], dtype=np.float32))
            cand_atb_dts.append(zeros_bbox)

        else:

            # if max iou is above a certain threshold
            if np.max(bbox_overlaps(types_bbox, j)) > thr:

                # take index of color bbox, store bbox and score
                idx = np.argmax(bbox_overlaps(types_bbox, j))
                atb_scores.append(atb_result[b][idx, -1:])
                cand_atb_dts.append(atb_result[b][idx, :])

            # if below the threshold set zeros_bbox and score to 0
            else:

                atb_scores.append(np.array([0], dtype=np.float32))
                cand_atb_dts.append(zeros_bbox)

    return np.argmax(np.stack(atb_scores)), np.stack(cand_atb_dts)[np.argmax(np.stack(atb_scores))]


def load_annotations(ann_file):
    """
    For loading the annotations in the middle format into the format used by MMDetection models.
    
    Args:
        - ann_file: annotation text file according to the middle format.
    """
    ann_list = mmcv.list_from_file(ann_file)

    data_infos = []
    for i, ann_line in enumerate(ann_list):
        if ann_line != '#':
            continue
        img_shape = ann_list[i + 2].split(' ')
        width = int(img_shape[0])
        height = int(img_shape[1])
        bbox_number = int(ann_list[i + 3])
        anns = ann_line.split(' ')
        bboxes = []
        labels = []
        for anns in ann_list[i + 4:i + 4 + bbox_number]:
            anns = np.fromstring(anns, sep=' ')
            bboxes.append([float(ann) for ann in anns[:4]])
            labels.append(int(anns[4]))

        data_infos.append(
            dict(
                filename=ann_list[i + 1],
                width=width,
                height=height,
                ann=dict(
                    bboxes=np.array(bboxes).astype(np.float32),
                    labels=np.array(labels).astype(np.int64))
            ))
    return data_infos


def get_ann_info(data_infos, idx):
    """
    Helper function to extract the value of the `ann` key from the dictionary in data_infos.
    
    Args:
        - data_infos: the data_info to perform the extraction on
        - idx: idx of the list to extract the value of the `ann` key
    """
    return data_infos[idx]['ann']

