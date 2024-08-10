"""
all utils files for data loading and usage

credit: https://github.com/faizan1234567/CKD-TransBTS/blob/main/utils.py
"""

import os
import SimpleITK as sitk
import numpy as np
import pandas as pd
import random
import torch
import torch.nn.functional as F
from pathlib import Path
import sys


def get_brats_folder(dataset_folder, mode):
    assert mode in ["train","train_val", "test"]
    if mode == "train":
        return os.path.join(dataset_folder, "brats2023", "train")
    elif mode == "train_val":
        return os.path.join(dataset_folder, "brats2023", "val")
    elif mode == "test" :
        return os.path.join(dataset_folder, "brats2023", "test")

def mkdir(folder):
    os.makedirs(folder, exist_ok=True)
    return folder

def save_best_model(args, model, name="best_model"):
    torch.save(model.state_dict(), f"{args.best_folder}/{name}.pkl")

def save_checkpoint(args, state, name="checkpont"):
    torch.save(state, f"{args.checkpoint_folder}/{name}.pth.tar")

def save_seg_csv(csv, args):
    try:
        val_metrics = pd.DataFrame.from_records(csv)
        columns = ['id', 'et_dice', 'tc_dice', 'wt_dice', 'et_hd', 'tc_hd', 'wt_hd', 'et_sens', 'tc_sens', 'wt_sens', 'et_spec', 'tc_spec', 'wt_spec']
        save_path = os.path.join(args.training.exp_name, "csv")
        os.makedirs(save_path, exist_ok= True)
        csv_path = os.path.join(save_path, "test_metrics.csv")
        val_metrics.to_csv(csv_path, index=False, columns=columns)
    except KeyboardInterrupt:
        print("Save CSV File Error!")

def load_nii(path):
    nii_file = sitk.GetArrayFromImage(sitk.ReadImage(str(path)))
    return nii_file

def listdir(path):
    files_list = os.listdir(path)
    files_list.sort()
    return files_list
    
def save_test_label(args, patient_id, predict):
    data_path = get_brats_folder(mode="test")
    ref_img = sitk.ReadImage(os.path.join(data_path, f"{patient_id}/{patient_id}_t1.nii.gz"))
    label_nii = sitk.GetImageFromArray(predict)
    label_nii.CopyInformation(ref_img)
    sitk.WriteImage(label_nii, os.path.join(args.label_folder, f"{patient_id}.nii.gz"))

class AverageMeter(object):
    def __init__(self, name, fmt):
        self.name = name
        self.fmt = fmt
        self.reset()
    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0
        self.avg = 0
    def update(self, val, n=1):
        if not np.isnan(val):
            self.val = val
            self.count += n
            self.sum += val * n
            self.avg = self.sum / self.count
    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def get_crop_slice(target_size,dim):
    if dim > target_size:
        crop_extent = dim - target_size
        left = random.randint(0, crop_extent)
        right = crop_extent - left
        return (left, dim - right)
    else:
        return (0, dim)

def get_left_right_idx_should_pad(target_size, dim):
    if dim >= target_size:
        return [False]
    else:
        pad_extent = target_size - dim
        left = random.randint(0, pad_extent)
        right = pad_extent - left
        return True, left, right
        
def pad_image_and_label(image, seg, target_size=(128, 128, 128)):
    c, z, y, x = image.shape
    pad_todos = [get_left_right_idx_should_pad(size, dim) for size, dim in zip(target_size, [z, y, x])]
    pad_list = [0, 0]
    for to_pad in pad_todos:
        if to_pad[0]:
            pad_list.insert(0, to_pad[1])
            pad_list.insert(0, to_pad[2])
        else:
            pad_list.insert(0, 0)
            pad_list.insert(0, 0)
    if np.sum(pad_list) != 0:
        image = F.pad(image, pad_list, 'constant')
    if seg is not None:
        if np.sum(pad_list) != 0:
            seg = F.pad(seg, pad_list,'constant')
        return image, seg, pad_list
    return image, seg, pad_list

def pad_or_crop_image(image, seg, target_size=(128, 128, 128)):
    c, z, y, x = image.shape
    z_slice, y_slice, x_slice = [get_crop_slice(target, dim) for target, dim in zip(target_size, (z, y, x))]
    crop_list = [z_slice, y_slice, x_slice]
    image = image[:, z_slice[0]:z_slice[1], y_slice[0]:y_slice[1], x_slice[0]:x_slice[1]]
    if seg is not None:
        seg = seg[:, z_slice[0]:z_slice[1], y_slice[0]:y_slice[1], x_slice[0]:x_slice[1]]
    image, seg, pad_list = pad_image_and_label(image, seg, target_size=target_size)
    return image, seg, pad_list, crop_list

def normalize(image):
    min_ = torch.min(image)
    max_ = torch.max(image)
    scale_ = max_ - min_
    image = (image - min_) / scale_
    return image

def minmax(image, low_perc=1, high_perc=99):
    non_zeros = image>0
    low, high = np.percentile(image[non_zeros], [low_perc, high_perc])
    image = torch.clip(image, low, high)
    image = normalize(image)
    return image
    
def cal_confuse(preds, targets, patient):
    assert preds.shape == targets.shape, "Preds and targets do not have the same size"
    labels = ["ET", "TC", "WT"]
    confuse_list = []
    for i, label in enumerate(labels):
        if torch.sum(targets[i]) == 0 and torch.sum(targets[i]==0):
            tp=tn=fp=fn=0
            sens=spec=1
        elif torch.sum(targets[i]) == 0:
            print(f'{patient} did not have {label}')
            sens = tp = fn = 0      
            tn = torch.sum(torch.logical_and(torch.logical_not(preds[i]), torch.logical_not(targets[i])))
            fp = torch.sum(torch.logical_and(preds[i], torch.logical_not(targets[i])))
            spec = tn / (tn + fp)
        else:
            tp = torch.sum(torch.logical_and(preds[i], targets[i]))
            tn = torch.sum(torch.logical_and(torch.logical_not(preds[i]), torch.logical_not(targets[i])))
            fp = torch.sum(torch.logical_and(preds[i], torch.logical_not(targets[i])))
            fn = torch.sum(torch.logical_and(torch.logical_not(preds[i]), targets[i]))

            sens = tp / (tp + fn)
            spec = tn / (tn + fp)
        confuse_list.append([sens, spec])
    return confuse_list

def cal_dice(predict, target, haussdor, dice):
    p_et = predict[0]
    p_tc = predict[1]
    p_wt = predict[2]
    t_et = target[0]
    t_tc = target[1]
    t_wt = target[2]
    p_et, p_tc, p_wt, t_et, t_tc, t_wt =  p_et.unsqueeze(0).unsqueeze(0), p_tc.unsqueeze(0).unsqueeze(0), p_wt.unsqueeze(0).unsqueeze(0), t_et.unsqueeze(0).unsqueeze(0), t_tc.unsqueeze(0).unsqueeze(0), t_wt.unsqueeze(0).unsqueeze(0)
    
    if torch.sum(p_et) != 0 and torch.sum(t_et) != 0:
        et_dice = float(dice(p_et, t_et).cpu().numpy())
        et_hd = float(haussdor(p_et, t_et).cpu().numpy())
    elif torch.sum(p_et) == 0 and torch.sum(t_et) == 0:
        et_dice =1
        et_hd = 0
    elif (torch.sum(p_et) == 0 and torch.sum(t_et) != 0) or (torch.sum(p_et) != 0 and torch.sum(t_et) == 0):
        et_dice =0
        et_hd = 347
    if torch.sum(p_tc) != 0 and torch.sum(t_tc) != 0:
        tc_dice = float(dice(p_tc, t_tc).cpu().numpy())
        tc_hd = float(haussdor(p_tc, t_tc).cpu().numpy())
    elif torch.sum(p_tc) == 0 and torch.sum(t_tc) == 0:
        tc_dice =1
        tc_hd = 0
    elif (torch.sum(p_tc) == 0 and torch.sum(t_tc) != 0) or (torch.sum(p_tc) != 0 and torch.sum(t_tc) == 0):
        tc_dice =0
        tc_hd = 347
    if torch.sum(p_wt) != 0 and torch.sum(t_wt) != 0:
        wt_dice = float(dice(p_wt, t_wt).cpu().numpy())
        wt_hd = float(haussdor(p_wt, t_wt).cpu().numpy())
    elif torch.sum(p_wt) == 0 and torch.sum(t_wt) == 0:
        wt_dice =1
        wt_hd = 0
    elif (torch.sum(p_wt) == 0 and torch.sum(t_wt) != 0) or (torch.sum(p_wt) != 0 and torch.sum(t_wt) == 0):
        wt_dice =0
        wt_hd = 347
    
    return [et_dice, tc_dice, wt_dice, et_hd, tc_hd, wt_hd]
