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
from configs import Config
from segment_3d import load_dataset
from segment_3d import inference
from segment_3d import model_loss_optim
from monai.data import DataLoader, decollate_batch
from monai.handlers.utils import from_engine
from monai.metrics import DiceMetric

def read_args():
    '''command line arguments for setting up 
    neccassary paths and params'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight', type = str, default = "", help = "weight \
        file path ")
    parser.add_argument('--workers', type = int, default=2,\
        help = "number of workers")
    parser.add_argument('--batch', type = int, default=1, \
        help= "batch size to load the dataset")
    opt = parser.parse_args()
    return opt


def add_paths(survival_df, name_mapping_df=None, t = 'test'):
    '''add paths to the csv file
    args:
    survival_df: dataframe
    name_mapping_df: dataframe
    t = str
    return 
    paths: list'''
    
    # rename dataframe on columns 
    if t!= "val" and t!= "test":
        name_mapping_df.rename({'BraTS_2020_subject_ID': 'Brats20ID'}, axis=1, inplace=True) 
        df = survival_df.merge(name_mapping_df, on="Brats20ID", how="right")
    else:
        df = survival_df
    paths = []
    temp_ids = []
    for _, row  in df.iterrows():
        id_ = row['Brats20ID']
        if t != "val" and t!= "test":
            phase = id_.split("_")[-2]
            if phase == 'Training':
                path = os.path.join(Config.train_root_dir, id_)
                if os.path.exists(path):
                    if len(os.listdir(path)) == 5:
                        p = os.path.join(Config.train_root_dir, id_)
                    else:
                        print('Not appending ID: {}'.format(path))
                        temp_ids.append(id_)
                else:  
                    temp_ids.append(id_)
        elif t == "val":
            path = os.path.join(Config.Testset.val_dir_path, id_)
            if os.path.exists(path):
                if len(os.listdir(path)) == 5:
                    paths.append(path)
                else:
                    print('Not appending ID: {}'.format(path))
                    temp_ids.append(id_)
            else:
                temp_ids.append(id_)
        else:
            path = os.path.join(Config.brats19_test_data, id_)
            if os.path.exists(path):
                if len(os.listdir(path)) == 5:
                    paths.append(path)
                else:
                    print('Not appending ID: {}'.format(path))
                    temp_ids.append(id_)
            else:
                temp_ids.append(id_)
    for id in temp_ids:
        df = df[df["Brats20ID"].str.contains(id) == False]
    df = df.reset_index()
    df.drop('index', inplace= True, axis=1)
    df['path'] = paths
    return df


def evaluate(model, weight, test_loader, post_transforms, 
             dice_metric, dice_metric_batch):
    '''to evaluate the model performance
    Args:
    model: deep learning model
    weight: learned params (path)
    test_loader: test data loader
    post_transforms: post transforms function
    dice_metric: dice evaluation metric
    dice_metric_batch: batch evaluation
    Return:
    None'''
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.load_state_dict(torch.load(weight,  map_location=torch.device('cpu')))
    model.eval()
    with torch.no_grad():
        for val_data in test_loader:
            val_inputs, val_labels = (
                    val_data["image"].to(device),
                    val_data["mask"].to(device))
            val_outputs = inference(val_inputs, model)
            val_outputs = [post_transforms(i) for i in decollate_batch(val_outputs)]
            # val_outputs, val_labels = from_engine(["pred", "mask"])(val_data)
            dice_metric(y_pred=val_outputs, y=val_labels)
            dice_metric_batch(y_pred=val_outputs, y=val_labels)

        metric_org = dice_metric.aggregate().item()
        metric_batch_org = dice_metric_batch.aggregate()

        dice_metric.reset()
        dice_metric_batch.reset()

    metric_tc, metric_wt, metric_et = metric_batch_org[0].item(), metric_batch_org[1].item(), metric_batch_org[2].item()

    print("Metric on original image spacing: ", metric_org)
    print(f"metric_tc: {metric_tc:.4f}")
    print(f"metric_wt: {metric_wt:.4f}")
    print(f"metric_et: {metric_et:.4f}") 

def main():
    args = read_args()
    print('configuring...')
    device, model, loss_function, optimizer, lr_scheduler, dice_metric, dice_metric_batch, post_trans = model_loss_optim(1, 1e-4, 1e-5)
    print('done\n')
    print('Loading test data... \n')
    survival_df_test = pd.read_csv(Config.brats19_test_survival_csv)
    df_test = add_paths(survival_df_test, t="test")
    test_dataset, test_loader = load_dataset(df_test,
                                             'val',
                                              False,
                                              args.workers,
                                              args.batch)
    print('done')
    print('Now evaluating models performance')
    evaluate(model, args.weight, test_loader, post_trans, dice_metric, dice_metric_batch)
    print('done')

if __name__ == '__main__':
    main()