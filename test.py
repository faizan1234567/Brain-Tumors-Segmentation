## Test set evaluation on brats19 dataset
#import neccassary packages...
#Author: Muhammad Faizan
#Date: 16.09.2022
#Time: 11.14pm
"""A script to evaluate the model performance"""
import argparse
import os
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import torch
from config.configs import Config
from DataLoader.dataset import BraTSDataset, get_dataloader
from monai.data import DataLoader, decollate_batch
from monai.handlers.utils import from_engine
from monai.metrics import DiceMetric
from utils.general import load_pretrained_model
from utils.meter import AverageMeter

def read_args():
    '''command line arguments for setting up 
    neccassary paths and params'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type = str, default = "", help = "weight \
        file path ")
    parser.add_argument('--fold', default= 0, type= int, 
                        help= "fold number for evaluation")
    parser.add_argument('--workers', type = int, default=2,\
        help = "number of workers")
    parser.add_argument('--batch', type = int, default=1, \
        help= "batch size to load the dataset")
    parser.add_argument('--json_file', type = str, default= "", \
                        help= "path to the data json file")
    parser.add_argument('--platform_changed', action= 'store_true', help= "running on other platfrom")
    opt = parser.parse_args()
    return opt


def evaluate(model,
              weights, 
              loader, 
              post_pred = None, 
              post_sigmoid = None,
              acc_func = None,
              model_inferer = None):
    '''to evaluate the model performance
    Parameters
    ----------:
    model: nn.Module
    weight: str
    loader: torch.utils.data.Dataset
    post_pred: monai.transforms.post.array.AsDiscrete
    post_sigmoid: monai.transforms.post.array.Activations
    acc_func: monai.metrics.meandice.DiceMetric 
    model_inferer: nn.Module
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_pretrained_model(model, state_path=weights)
    model.eval()
    model.to(device)
    tic = time.time()
    run_acc = AverageMeter()
    with torch.no_grad():
        for index, batch_data in enumerate(loader):
            prediction_lists = decollate_batch(model_inferer(batch_data["image"].to(device)))
            masks = decollate_batch(batch_data["label"].to(device)) 
            predictions = [post_pred(post_sigmoid(prediction)) for prediction in prediction_lists]
            acc_func.reset()
            acc_func(y_pred = predictions, y = masks)
            acc, not_nans = acc_func.aggregate()
            run_acc.update(acc.cpu().numpy(), n = not_nans.cpu().numpy())
            dice_tc = run_acc.avg[0]
            dice_wt = run_acc.avg[1]
            dice_et = run_acc.avg[2]
            print(
                "Val  {}/{}".format(index, len(loader)),
                ", dice_tc:",
                dice_tc,
                ", dice_wt:",
                dice_wt,
                ", dice_et:",
                dice_et,
                ", time {:.2f}s".format(time.time() - tic),
            )
            tic = time.time()
    return run_acc.avg


def main():
    """
    Function that handles everything..
    """
    args = read_args()
    print('Configuring...')
    if args.platform_changed:
        data_csv_file_path = Config.newGlobalConfigs.OtherPC.path_to_csv
        data_json_file = Config.newGlobalConfigs.OtherPC.json_file
        data_dir = Config.newGlobalConfigs.OtherPC.train_root_dir
        data_json_file = Config.newGlobalConfigs.OtherPC.json_file
    else:
        data_csv_file_path = Config.newGlobalConfigs.path_to_csv
        data_json_file = Config.newGlobalConfigs.json_file
        data_dir = Config.newGlobalConfigs.train_root_dir
        data_json_file = Config.newGlobalConfigs.json_file

    model = Config.newGlobalConfigs.swinUNetCongis.training_cofigs.model
    weights = args.weights
    batch_size = args.batch
    wokers = args.workers
    fold = args.fold
    post_pred = Config.newGlobalConfigs.swinUNetCongis.training_cofigs.post_pred
    post_sigmoid = Config.newGlobalConfigs.swinUNetCongis.training_cofigs.post_simgoid
    acc_func = Config.newGlobalConfigs.swinUNetCongis.training_cofigs.dice_acc
    model_inferer = Config.newGlobalConfigs.swinUNetCongis.training_cofigs.model_inferer
    if args.json_file:
        data_json_file = args.json_file
    print('Configured. Now Loading dataset...')
    test_loader = get_dataloader(BraTSDataset,
                                 path_to_csv= data_csv_file_path,
                                 phase = 'val',
                                 batch_size= batch_size,
                                 num_workers= wokers,
                                 json_file=data_json_file,
                                 fold = fold,
                                 train_dir= data_dir)
    
    print('The dataset loaded. \n')
    print(f'Dataset size: {len(test_loader)}')
    print('Now starting evaluation on the test set.')
    mean_dice =evaluate(model= model,
             weights=weights,
             loader= test_loader,
             post_pred= post_pred,
             post_sigmoid= post_sigmoid,
             acc_func= acc_func,
             model_inferer= model_inferer)
    
    print("Mean dice on the test set: ", mean_dice)
    
    print('Completed sucessfully!!')
    
if __name__ == '__main__':
    main()