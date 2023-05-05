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
import nibabel as nib
import tqdm as tqdm
from utils.meter import AverageMeter


import monai
import torch
import torch.nn as nn

device = "cuda" if torch.cuda.is_available else "cpu"

def train_epoch(model, loader, optimizer, loss_func, epoch, max_epochs = 100):
    """train the model for epoch on MRI image and given ground truth labels
    using set of arguments
    
    Parameters
    ----------
    model: nn.Module
    loader: torch.utils.data.Dataset
    optimizer: 
    loss_func: 
    epoch: int"""
    model.train() 
    tic = time.time()
    run_loss = AverageMeter()
    for index, batch_data in enumerate(loader):
        image_data, mask = batch_data["image"].to(device), batch_data["labels"].to(device)
        logits = model(image_data)
        loss = loss_func(logits, mask)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        run_loss.update(loss.item(), n = batch_data.shape[0])
        print(
            "Epoch {}/{} {}/{}".format(epoch, max_epochs, index, len(loader)),
            "loss: {:.4f}".format(run_loss.avg),
            "time {:.2f}s".format(time.time() - tic))
        print()
        tic = time.time()
    return run_loss.avg




