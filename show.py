"""
script to show the image and label, or image with labeled mask
-------------------------------------------------------------

Author: Muhammad Faizan
Date: 13 May 2023
Copywrite (c) Muhammad Faizan
---------------------------------------------------------------
"""

import matplotlib.pyplot as plt
import logging
import numpy as np
import imageio
import argparse


import torch
import monai
from DataLoader.dataset import BraTSDataset, get_dataloader
from config.configs import Config
import yaml
from utils.visualizer import visualize_abnormal_area, get_labelled_image, visualize_data_gif
from utils.general import visualize_data_sample

#configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s: %(name)s: %(message)s")
file_handler = logging.FileHandler(filename= "logs/show.log")
stream_handler = logging.StreamHandler()
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)

# get command line arguments i.e. from a user
def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_file', type = str, help= "dataset split file")
    parser.add_argument('--fold', type= int, default= 0, help= "folder number")
    parser.add_argument('--phase', type = str, default= 'val', help= "validation or training.")
    parser.add_argument('--save', type = str, default= "runs/", help= "results save directory")
    parser.add_argument('--get_abnormal_area', action= 'store_true', help = 'get full abnormal are')
    parser.add_argument('--visualize_data_gif', action= 'store_true', help= "visulize data gif, and create a gif file")
    parser.add_argument('--visualize_data_sample', action = 'store_true', help= "visualize one sample")
    parser.add_argument("--config", type = str, default="config/configs.yaml", help = "path to configs file")
    opt = parser.parse_args()
    return opt


def show_result(args: argparse.Namespace):
    """
    Visualize labelled brain scan on a patient case, three options are available
    1 - create brain scan slices and label them
    2 - create a .gif format file to visualize part of brain (labelled)
    3 - visualize a scan with it's label in a subplot format

    Parameters
    ----------
    args: argparse.Namespace()
    """
    # load a config file for loading configuration settings
    with open(args.config, 'r') as file:
        configs = yaml.safe_load(file)

    full_paths = configs["config"]['full_paths']

    try:
        json_file = args.json_file
    except AttributeError:
        logger.info(f"{args.json_file} not passed, setting file from the configs")
        json_file = full_paths["json_file"]
    json_file = full_paths["json_file"]
    phase = args.phase
   
    # load the dataset, a sample for visualization
    data = get_dataloader(dataset = BraTSDataset, 
                          path_to_csv = full_paths["dataset_file"], 
                          phase= phase, batch_size= 1, num_workers= 2, 
                          json_file=json_file,
                          is_process_mask= False)
    
    # iterate over a batch to get a batch of examples
    batch = next(iter(data))
    image, label = batch["image"], batch['label']
    logger.info('visualizing an image with label')

    # visualize selected option
    if args.get_abnormal_area:
        visualize_abnormal_area(image, label)
    elif args.visualize_data_gif:
        labelled_img = get_labelled_image(image, label)
        visualize_data_gif(labelled_img)
    elif args.visualize_data_sample:
        visualize_data_sample(full_paths['test_patient'], 
                              configs["config"]['dataset']['a_test_patient'])
    else:
        logger.info('No option selected')


if __name__ == "__main__":
    # run everything here..
    args = read_args()
    show_result(args)
    print('Done!!!')
    



