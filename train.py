"""
train BraTS21 Tumor segmentation model
Author: Muhammad Faizan
Date: 5 May 2023
National University of Sciences and Technology Islamabad, Pakistan

"""
import os
import copy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import sys
import time
import argparse
import nibabel as nib
import tqdm as tqdm
from utils.meter import AverageMeter
from config.configs import *
from utils.general import save_checkpoint, load_pretrained_model, resume_training
from DataLoader.dataset import BraTSDataset, get_dataloader

import monai
from monai.data import create_test_image_3d, Dataset, DataLoader, decollate_batch
import torch
import torch.nn as nn

def read_args():
    """read commmand line arguments"""
    parser = argparse.ArgumentParser(description="command line args")
    parser.add_argument('--data', default= "", type= str, help= "dataset root directory path")
    parser.add_argument('--fold', default= 0, type = int, help="folder name")
    parser.add_argument('--json_file', default= "", type = str, help ="path to json file")
    parser.add_argument('--batch', default=1, type= int, help= "batch size")
    parser.add_argument('--img_roi', default=128, type = int, help = 'image roi size')
    parser.add_argument('--val_every', default= 2, type = int, help= "validate every 2 epochs")
    parser.add_argument('--max_epochs', default= 100, type= int, help= "maximum number of epoch to train")
    parser.add_argument('--workers', default=2, type = int, help= "Number of data loading workers")
    parser.add_argument('--pretrained_model', default= "", type = str, help = "path to pretraiend model")
    parser.add_argument('--pretrained', action= 'store_true', help= "use pretrained weights.")
    parser.add_argument('--resume', action= 'store_true', help="starting training from the saved ckpt.")
    parser.add_argument('--platform_changed', action='store_true', help="pc changed, so that dataset dir has been set accordingly")
    opt = parser.parse_args()
    return opt

def train_epoch(model, loader, optimizer, loss_func, epoch, max_epochs = 100):
    """
    train the model for epoch on MRI image and given ground truth labels
    using set of arguments
    
    Parameters
    ----------
    model: nn.Module
    loader: torch.utils.data.Dataset
    optimizer: torch.optim.adamw.AdamW
    loss_func: monai.losses.dice.DiceLoss
    epoch: int
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train() 
    tic = time.time()
    run_loss = AverageMeter()
    for index, batch_data in enumerate(loader):
        logits = model(batch_data["image"].to(device))
        loss = loss_func(logits, batch_data["label"].to(device))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        run_loss.update(loss.item(), n = batch_data["image"].shape[0])
        print(
            "Epoch {}/{} {}/{}".format(epoch, max_epochs, index, len(loader)),
            "loss: {:.4f}".format(run_loss.avg),
            "time {:.2f}s".format(time.time() - tic))
        print()
        tic = time.time()
    return run_loss.avg


def val(model, loader, acc_func,
        max_epochs = None, epoch = None, model_inferer = None,
        post_sigmoid = None, post_pred = None):
    """
    Validation phase
    use model and validation dataset to validate the model performance on 
    validation dataset.

    Parameters
    ----------
    model: nn.Module
    loader: torch.util.data.Dataset
    acc_func: monai.metrics.meandice.DiceMetric 
    num_epochs: int
    epochs: int
    model_inferer: nn.Module
    post_sigmoid: monai.transforms.post.array.Activations
    post_pred:monai.transforms.post.array.AsDiscrete
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    tic = time.time()
    run_acc = AverageMeter()
    with torch.no_grad():
        for index, batch_data in enumerate(loader):
            logits = model_inferer(batch_data["image"].to(device))
            masks = decollate_batch(batch_data["label"].to(device)) 
            prediction_lists = decollate_batch(logits)
            predictions = [post_pred(post_sigmoid(prediction)) for prediction in prediction_lists]
            acc_func.reset()
            acc_func(y_pred = predictions, y = masks)
            acc, not_nans = acc_func.aggregate()
            run_acc.update(acc.cpu().numpy(), n = not_nans.cpu().numpy())
            dice_tc = run_acc.avg[0]
            dice_wt = run_acc.avg[1]
            dice_et = run_acc.avg[2]
            print(
                "Val {}/{} {}/{}".format(epoch, max_epochs, index, len(loader)),
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

def save_data(training_loss,
              et, wt, tc,
              val_mean_acc,
              val_losses,
              training_dices,
              epochs):
    """save the training data for later use
    
    Parameters
    ----------
    training_loss: list
    et: list
    wt: list
    tc: list
    val_mean_acc: list
    val_losses: list
    tarining_dices: list,
    epochs: list
    """
    data = {}
    NAMES = ["training_loss", "WT", "ET", "TC", "mean_dice", "epochs"]
    data_lists = [training_loss, wt, et, tc, val_mean_acc, epochs]
    for i in range(len(NAMES)):
        data[f"{NAMES[i]}"] = data_lists[i]
    data_df = pd.DataFrame(data)
    data_df.to_csv('training_data.csv')
    return data

def trainer(model,
            train_loader,
            val_loader,
            optimizer,
            loss_func,
            acc_func,
            scheduler,
            max_epochs = 100,
            model_inferer = None,
            start_epoch = 0,
            post_sigmoid = None,
            post_pred = None,
            val_every = 2):
    """
    train and validate the model

    model: nn.Module
    train_loader: torch.utils.data.Dataset
    val_loader: torch.utils.data.Dataset
    optimizer: torch.optim
    loss_func: monai.losses.dice.DiceLoss
    acc_func:  monai.metrics.meandice.DiceMetric 
    schedular: torch.optim.lr_scheduler.CosineAnnealingLR
    max_epochs: int
    model_inferer: nn.Module
    start_epoch: int
    post_sigmoid: monai.transforms.post.array.Activations
    post_pred: monai.transforms.post.array.AsDiscrete
    """
    val_acc_max = 0
    dices_tc = []
    dices_wt = []
    dices_et = []
    mean_dices = []
    epoch_losses = [] # training loss
    train_epochs = []
    for epoch in range(start_epoch, max_epochs):
        print()
        print(time.ctime(), "Epoch: ", epoch)
        epoch_time = time.time()
        training_loss = train_epoch(model=model,
                                    loader= train_loader,
                                    optimizer=optimizer,
                                    loss_func= loss_func,
                                    epoch= epoch,
                                    max_epochs=max_epochs)
        print(
            "Final training  {}/{}".format(epoch + 1, max_epochs - 1),
            "loss: {:.4f}".format(training_loss),
            "time {:.2f}s".format(time.time() - epoch_time),
        )

        if epoch % val_every == 0 or epoch == 0:
            epoch_losses.append(training_loss)
            train_epochs.append(int(epoch))
            val_epoch_time = time.time()
            val_acc =  val(model= model,
                          loader= val_loader,
                          acc_func= acc_func,
                          max_epochs= max_epochs,
                          epoch = epoch,
                          model_inferer= model_inferer,
                          post_sigmoid=post_sigmoid,
                          post_pred=post_pred)
            dice_tc = val_acc[0]
            dice_wt = val_acc[1]
            dice_et = val_acc[2]
            val_mean_acc = np.mean(val_acc)
            print(
                "Final validation stats {}/{}".format(epoch + 1, max_epochs - 1),
                ", dice_tc:",
                dice_tc,
                ", dice_wt:",
                dice_wt,
                ", dice_et:",
                dice_et,
                ", Dice_Avg:",
                val_mean_acc,
                ", time {:.2f}s".format(time.time() - epoch_time),
            )
            dices_tc.append(dice_tc)
            dices_et.append(dice_et)
            dices_wt.append(dices_wt)
            mean_dices.append(val_mean_acc)
            if val_mean_acc > val_acc_max:
                print("new best ({:.6f} --> {:.6f}). ".format(val_acc_max, val_mean_acc))
                val_acc_max = val_mean_acc
                save_checkpoint(model=model,
                                epoch= epoch,
                                best_acc=val_acc_max)
            scheduler.step()
    print("Training Finished !, Best Accuracy: ", val_acc_max)
    save_data(training_loss=training_loss,
              et= dices_et,
              wt= dices_wt,
              tc=dices_tc,
              val_mean_acc=mean_dices,
              epochs=train_epochs)
    
    return (
        val_acc_max,
        dices_tc,
        dices_wt,
        dices_et,
        mean_dices,
        training_loss,
        train_epochs)

def run(args, model,
        loss_func,
        acc_func,
        optimizer,
        train_loader,
        val_loader,
        scheduler,
        model_inferer = None,
        post_sigmoid = None, 
        post_pred = None,
        max_epochs = 100,
        start_epoch = 0,
        val_every = 2
        ):
    '''Now train the model
    
    Parameters
    ----------
    args: argparse.parser
    model: nn.Module
    acc_func:  monai.metrics.meandice.DiceMetric
    loss_func: monai.losses.dice.DiceLoss
    optimizer: torch.optim.adamw.AdamW
    train_loader: torch.utils.data.Dataset
    val_loader: torch.utils.data.Dataset
    schedular:  torch.optim.lr_scheduler.CosineAnnealingLR
    model_inferer: nn.Module
    post_sigmoid: monai.transforms.post.array.Activations
    post_pred:monai.transforms.post.array.AsDiscrete
    max_epochs: int
    start_epoch: int
    val_every: int
    '''
    if args.pretrained:
        print('Loading a pretrained model')
        print()
        model = load_pretrained_model(model, args.pretrained_model)
    if args.resume:
        print('Resuming training...')
        model = resume_training(model, args.pretrained_model)
    elif args.pretrained:
        print("Using a pretrained model...")
    else:
        print('Trainig from scrath!')

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print()
    print("Total parameters count", total_params)

    (
    val_mean_dice_max,
    dices_tc,
    dices_wt,
    dices_et,
    dices_mean,
    train_losses,
    train_epochs,
    ) = trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_func=loss_func,
        acc_func=acc_func,
        scheduler=scheduler,
        model_inferer=model_inferer,
        start_epoch=start_epoch,
        post_sigmoid=post_sigmoid,
        post_pred=post_pred,
    )
    print()
    print(f"train completed, best average dice: {val_mean_dice_max:.4f} ")
    return (val_mean_dice_max, 
            dices_tc,
            dices_wt,
            dices_et,
            dices_mean,
            train_losses,
            train_epochs)


if __name__ == "__main__":
    print("Configuring Variables..")
    print()
    start_epoch = 0
    torch.backends.cudnn.benchmark = True
    args = read_args()
    post_pred = Config.newGlobalConfigs.swinUNetCongis.training_cofigs.post_pred
    post_sigmoid = Config.newGlobalConfigs.swinUNetCongis.training_cofigs.post_simgoid
    model = Config.newGlobalConfigs.swinUNetCongis.training_cofigs.model
    model_inferer = Config.newGlobalConfigs.swinUNetCongis.training_cofigs.model_inferer
    val_every = args.val_every
    loss_func = Config.newGlobalConfigs.swinUNetCongis.training_cofigs.dice_loss
    acc_func = Config.newGlobalConfigs.swinUNetCongis.training_cofigs.dice_acc
    optimizer = Config.newGlobalConfigs.swinUNetCongis.training_cofigs.optimizer
    scheduler = Config.newGlobalConfigs.swinUNetCongis.training_cofigs.scheduler
    max_epochs = args.max_epochs
    dataset_info_csv = Config.newGlobalConfigs.path_to_csv
    batch_size = args.batch
    num_workers = args.workers
    if args.platform_changed:
        train_dir = Config.newGlobalConfigs.OtherPC.train_root_dir
        dataset_info_csv = Config.newGlobalConfigs.OtherPC.path_to_csv
        json_file = Config.newGlobalConfigs.OtherPC.json_file
    else:
        train_dir = Config.newGlobalConfigs.train_root_dir
        dataset_info_csv = Config.newGlobalConfigs.path_to_csv
        json_file = Config.newGlobalConfigs.json_file

    print("Configured. Now Loading the dataset...\n")
    train_loader = get_dataloader(BraTSDataset, 
                                  dataset_info_csv, 
                                  phase = "train",
                                  batch_size= batch_size, 
                                  num_workers=num_workers,
                                  json_file=json_file,
                                  fold=args.fold,
                                  train_dir = train_dir)
    
    val_loader = get_dataloader(BraTSDataset, 
                                dataset_info_csv, 
                                phase= "val", 
                                batch_size=batch_size,  
                                num_workers=num_workers,
                                json_file=json_file,
                                fold=args.fold, 
                                train_dir= train_dir)
    print('starting training...')
    print('--'* 40)
    run(args, model=model,
        loss_func= loss_func,
        acc_func= acc_func,
        optimizer= optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        scheduler=scheduler,
        model_inferer=model_inferer,
        post_sigmoid=post_sigmoid,
        post_pred=post_pred,
        max_epochs=max_epochs,
        start_epoch=start_epoch,
        val_every=val_every)
    print('Done!!!')
