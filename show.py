"""
script to show the image and label, or image with labeled mask


Author: Muhammad Faizan
Date: 13 May 2023


"""

import matplotlib.pyplot as plt
import logging
import numpy as np
import imageio
import os
import sys
import random
import argparse


import torch
import monai
from DataLoader.dataset import BraTSDataset, get_dataloader
from config.configs import Config
from utils.visualizer import visualize_abnormal_area, get_labelled_image, visualize_data_gif
from utils.general import visualize_data_sample

#configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s: %(name)s: %(message)s")
file_handler = logging.FileHandler(filename= "show.log")
stream_handler = logging.StreamHandler()
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)

def read_args():
    """read command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_file', type = str, default= Config.newGlobalConfigs.json_file, help= "dataset split file")
    parser.add_argument('--fold', type= int, default= 0, help= "folder number")
    parser.add_argument('--phase', type = str, default= 'val', help= "validation or training.")
    parser.add_argument('--save', type = str, default= "runs/", help= "results save directory")
    parser.add_argument('--get_abnormal_area', action= 'store_true', help = 'get full abnormal are')
    parser.add_argument('--visualize_data_gif', action= 'store_true', help= "visulize data gif, and create a gif file")
    parser.add_argument('--visualize_data_sample', action = 'store_true', help= "visualize one sample")
    opt = parser.parse_args()
    return opt


def show_result(args):
    """
    show result as per the requirments
    Parameters
    ----------
    args: argparse.parser
    """
    if args.json_file:
        json_file = args.json_file
    else:
        json_file = Config.newGlobalConfigs.json_file
    phase = args.phase
    data = get_dataloader(BraTSDataset, 
                          Config.newGlobalConfigs.path_to_csv, 
                          phase, 1, 2, 
                          json_file=json_file,
                          is_process_mask= False)
    batch = next(iter(data))
    image, label = batch["image"], batch['label']
    print('visualizing an image with label')
    if args.get_abnormal_area:
        visualize_abnormal_area(image, label)
    elif args.visualize_data_gif:
        labelled_img = get_labelled_image(image, label)
        visualize_data_gif(labelled_img)
    elif args.visualize_data_sample:
        visualize_data_sample(Config.newGlobalConfigs.full_patient_path, 
                              Config.newGlobalConfigs.a_test_patient)
    else:
        print('No option selected')


if __name__ == "__main__":
    args = read_args()
    show_result(args)
    print('Done!!!')
    



