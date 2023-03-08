from utils import *

import os
import glob
from random import randint
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Process data to middle format')
    parser.add_argument('--data-dir', default ='/local_data/local_data/I-RAVEN/',help='directory with the folders containing the figure configurations.')
    parser.add_argument('--label-attrib', nargs="+", default='types', help='the label attribute')
    parser.add_argument('--save-dir', default='/local_data/local_data/midformat', help='the dir to save dataset folder.')
    parser.add_argument('--train', type=int, default=60, help='number of .npz training samples.')
    parser.add_argument('--val', type=int, default=20, help='number of .npz validation samples.')
    parser.add_argument('--test', type=int, default=20, help='number of .npz test samples.')
    parser.add_argument('--bbox-format', type=str, default='xy', help='format of bbox, either `xy` or `wh`.')
    parser.add_argument('--seed', type=str, default=0, help='seed for random shuffle of samples.')
    parser.add_argument('--attrib-dir', type=str, default=None, help='')
    parser.add_argument('--model-dir', type=str, default='./work_dirs/perception-types--D06-01-2022--T09-23-45/', help='')
    parser.add_argument('--gpu-ids', type=int, default=None, help='')
    parser.add_argument('--batch-size', type=int, default=16, help='')
    
    args = parser.parse_args()
    return args
    
def main():
    args = parse_args()
        
    print('Processing {} attribute data for object detection task...'.format(', '.join(args.label_attrib)))
    process_det_midformat(args.data_dir,
                      args.save_dir,
                      split = (args.train,args.val,args.test),
                      atb = args.label_attrib,
                      fig_config = None,
                      df = args.bbox_format,
                      seed = args.seed)
    print('Data saved to {}'.format(args.save_dir))


if __name__ == '__main__':
    main()
