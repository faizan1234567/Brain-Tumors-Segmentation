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
import matplotlib.pyplot as plt
import torch
from config.configs import Config
from DataLoader.dataset import BraTSDataset, get_dataloader
from monai.data import DataLoader, decollate_batch
from monai.handlers.utils import from_engine
from monai.metrics import DiceMetric
from utils.general import load_pretrained_model
from train import val



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
    model = load_pretrained_model(model, state_path=weights)
    mean_dice, test_loss = val(model=model,
                               loader= loader,
                               acc_func=acc_func,
                               model_inferer=model_inferer,
                               post_pred= post_pred,
                               post_sigmoid= post_sigmoid,
                               )
    

    print("Mean dice on the test set: ", mean_dice)
    print('Loss on the test set: {}'.format(test_loss))

def main():
    """
    Function that handles everything..
    """
    args = read_args()
    print('Configuring...')
    model = Config.newGlobalConfigs.swinUNetCongis.training_cofigs.model
    weights = args.weights
    data_csv_file_path = Config.newGlobalConfigs.OtherPC.path_to_csv
    batch_size = args.batch
    wokers = args.workers
    data_json_file = Config.newGlobalConfigs.OtherPC.json_file
    fold = args.fold
    post_pred = Config.newGlobalConfigs.swinUNetCongis.training_cofigs.post_pred
    post_sigmoid = Config.newGlobalConfigs.swinUNetCongis.training_cofigs.post_simgoid
    acc_func = Config.newGlobalConfigs.swinUNetCongis.training_cofigs.dice_acc
    model_inferer = Config.newGlobalConfigs.swinUNetCongis.training_cofigs.model_inferer
    data_dir = Config.newGlobalConfigs.OtherPC.train_root_dir
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
    evaluate(model= model,
             weights=weights,
             loader= test_loader,
             post_pred= post_pred,
             post_sigmoid= post_sigmoid,
             acc_func= acc_func,
             model_inferer= model_inferer)
    
    print('Completed sucessfully!!')
    
if __name__ == '__main__':
    main()