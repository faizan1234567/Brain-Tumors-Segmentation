from ast import increment_lineno
import logging
import sys
import argparse
import os
import matplotlib
import numpy as np
from tensorboard import notebook
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from configs import Config

import os
import shutil
import tempfile
import time
import matplotlib.pyplot as plt
import numpy as np
from BratsCustom import BratsDataset20
from monai.apps import DecathlonDataset
from monai.config import print_config
from monai.data import DataLoader, decollate_batch
from monai.handlers.utils import from_engine
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.networks.nets import SegResNet
from monai.transforms import (
    Activations,
    Activationsd,
    AsDiscrete,
    AsDiscreted,
    Compose,
    Invertd,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    Spacingd,
    EnsureChannelFirstd,
    EnsureTyped,
    EnsureType,
)
from monai.utils import set_determinism



def read_args():
    '''reading command line arguments, such 
    as hyper parameters'''
    parser = argparse.ArgumentParser(description= "parsing command line args for\
                                                training a segmentation model on medical\
                                                imaging dataset")

    parser.add_argument('--epochs', type = int, help= "number of epochs to run the model for...")
    parser.add_argument('--lr', type = float, help = "learning rate")
    parser.add_argument('--weight_decay', type = float, help= "weight decay")
    parser.add_argument('--batch', type = int, help= "batch size")
    parser.add_argument('--data_dir', type =str, help= "data directory to put all the dataset..")
    parser.add_argument('--name', type = str, help= "name of the data dir")
    parser.add_argument('--workers', type = int, help= 'number of workers')
    parser.add_argument('--resume', action="store_true", help="set to resume the training")
    parser.add_argument('--weights', default= "", type = str, help= "path to weight file in .pth extension")
    return parser.parse_args()

def root_directory(data, name):
    '''creat a temp root directory
    Args:
    data: path(str) root directory path
    name: name of the dataset directory under root dir
    return:
    data_dir: path(str) path to put all the data in ..
    '''
    global directory
    directory = os.environ.get("MONAI_DATA_DIRECTORY")
    # os.chdir(data)
    data_dir = os.path.join(data, name)
    if not os.path.exists(data_dir) and directory is None:
        os.mkdir(data_dir)
    else:
        print(f"{data_dir } already exists!!")
    # root_dir = tempfile.mkdtemp() if directory is None else directory
    # print(f'root directory: {root_dir}') 
    return data_dir

class ConvertToMultiChannelBasedforBratsClassesd(MapTransform):
    """
    Convert labels to multi channels based on brats classes:
    label 1 is the peritumoral edema
    label 2 is the GD-enhancing tumor
    label 3 is the necrotic and non-enhancing tumor core
    The possible classes are TC (Tumor core), WT (Whole tumor)
    and ET (Enhancing tumor).

    """

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            # merge label 2 and label 3 to construct TC
            result.append(np.logical_or(d[key] == 2, d[key] == 3))
            # merge labels 1, 2 and 3 to construct WT
            result.append(
                np.logical_or(
                    np.logical_or(d[key] == 2, d[key] == 3), d[key] == 1
                )
            )
            # label 2 is ET
            result.append(d[key] == 2)
            d[key] = np.stack(result, axis=0).astype(np.float32)
        return d

def add_paths(survival_df, name_mapping_df=None, t = 'train'):
    '''add paths to the csv file
    args:
    survival_df: dataframe
    name_mapping_df: dataframe
    t = str
    return 
    paths: list'''
    
    # rename dataframe on columns 
    if t!= "val":
      if t == "test":
         survival_df.rename({"BraTS20ID": "Brats20ID"}, axis=1, inplace=True)
      name_mapping_df.rename({'BraTS_2020_subject_ID': 'Brats20ID'}, axis=1, inplace=True) 
      df = survival_df.merge(name_mapping_df, on="Brats20ID", how="right")
    else:
        df = survival_df
    paths = []
    temp_ids = []
    for _, row  in df.iterrows():
        #BraTS20_Training_020
        id_ = row['Brats20ID']
        if t != "val":
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
            else:
                path = os.path.join(Config.test_root_dir, id_)
                if os.path.exists(path):
                    if len(os.listdir(path)) == 4:
                        p = os.path.join(Config.test_root_dir, id_)
                    else:
                        print('Not appending ID: {}'.format(path))
                        temp_ids.append(id_)
                else:
                    temp_ids.append(id_)
            paths.append(p)  
        else:
            path = os.path.join(Config.Testset.val_dir_path, id_)
            if os.path.exists(path):
                if len(os.listdir(path)) == 5:
                    paths.append(path)
                else:
                    print('Not appending ID: {}'.format(path))
                    temp_ids.append(id_)
            else:
    
                temp_ids.append(id_)
    if len(temp_ids)>0:
        for id in temp_ids:
            df = df[df["Brats20ID"].str.contains(id) == False]

    if t == 'train':
        #error files
        ids = ['BraTS20_Training_325', 'BraTS20_Training_266', 'BraTS20_Training_195']
        for id in ids:
            df = df[df["Brats20ID"].str.contains(id) == False]
    
    df = df.reset_index()
    df.drop('index', inplace= True, axis=1)
    paths = list(set(paths))
    df['path'] = paths
    df = df.drop_duplicates()
    print("paths length: {}".format(len(paths)))
    return df


def load_dataset(df, data = 'train', resize = False, num_workers =2, batch_size=1):
    '''load the dataset the Brats-2020 dataset from the local directory
    and write the dataset in a folder.
       Args:
       df: pd.dataframe
       data: str (type of data, i.e train, val, test)
       resize: bool defalut -> False (used to resize samples)
       num_workers: int, number of workers to load the dataset from data dir
       batch_size: int, mini-batch size
       
       Return:
       dataset: object (iterable)
       datalaoder: object (iterable) 
       '''
    # here we don't cache any data in case out of memory issue
    if data =='train':
        dataset = BratsDataset20(df, data, is_resize= resize, mask_label=True)
        dataset_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    else:
        dataset = BratsDataset20(df, data, is_resize= resize)
        dataset_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return dataset, dataset_loader

def model_loss_optim(epochs, lr, weight_decay):
    """craete model, loss, and optimizer for training
    args:
    epochs: num of epochs to train for
    lr: learning rate
    weight_decay: weight decay of the optimization algortithm"""
    max_epochs = epochs
    val_interval = 1
    # VAL_AMP/ = True

     # standard PyTorch program style: create SegResNet, DiceLoss and Adam optimizer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SegResNet(
    blocks_down=[1, 2, 2, 4],
    blocks_up=[1, 1, 1],
    init_filters=16,
    in_channels=4,
    out_channels=3,
    dropout_prob=0.2,).to(device)

    loss_function = DiceLoss(smooth_nr=0, smooth_dr=1e-5, squared_pred=True, to_onehot_y=False, sigmoid=True)
    optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    dice_metric = DiceMetric(include_background=True, reduction="mean")
    dice_metric_batch = DiceMetric(include_background=True, reduction="mean_batch")

    post_trans = Compose(
    [EnsureType(), Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    return device, model, loss_function, optimizer, lr_scheduler, dice_metric, dice_metric_batch, post_trans


# define inference method
def inference(input, model):
    VAL_AMP = True
    def _compute(input):
        return sliding_window_inference(
            inputs=input,
            roi_size=(240, 240, 160),
            # roi_size = (240, 240),
            sw_batch_size=1,
            predictor=model,
            overlap=0.5,
        )

    if VAL_AMP:
        with torch.cuda.amp.autocast():
            return _compute(input)
    else:
        return _compute(input)



# scaler = torch.cuda.amp.GradScaler()
# # enable cu# use amp to accelerate trainingDNN benchmark
# torch.backends.cudnn.benchmark = True

def visualize(validation_ds):
    '''check shape and visualize dataset'''
    # pick one image from DecathlonDataset to visualize and check the 4 channels
    val_data_example = validation_ds[2]
    print(f"image shape: {val_data_example['image'].shape}")
    plt.figure("image", (24, 6))
    for i in range(4):
        plt.subplot(1, 4, i + 1)
        plt.title(f"image channel {i}")
        plt.imshow(val_data_example["image"][i, :, :, 60].detach().cpu(), cmap="gray")
    plt.savefig('image_visualize.png')
    plt.show()
    plt.close()
    
   # also visualize the 3 channels label corresponding to this image
    print(f"label shape: {val_data_example['mask'].shape}")
    plt.figure("label", (18, 6))
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.title(f"label channel {i}")
        plt.imshow(val_data_example["mask"][i, :, :, 60].detach().cpu())
    plt.savefig('label_visualize.png')
    plt.show()
    

def train(max_epochs, model, train_loader, device, optimizer, loss_function, train_ds,lr_scheduler, val_loader,dice_metric, dice_metric_batch, root_dir, scaler, post_trans):

    '''train the model'''
    val_interval = 1
    best_metric = -1
    best_metric_epoch = -1
    best_metrics_epochs_and_time = [[], [], []]
    epoch_loss_values = []
    metric_values = []
    metric_values_tc = []
    metric_values_wt = []
    metric_values_et = []

    total_start = time.time()
    for epoch in range(max_epochs):
        epoch_start = time.time()
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step_start = time.time()
            step += 1
            inputs, labels = (
                batch_data["image"].to(device),
                batch_data["mask"].to(device),)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
            print(
                f"{step}/{len(train_ds) // train_loader.batch_size}"
                f", train_loss: {loss.item():.4f}"
                f", step time: {(time.time() - step_start):.4f}")
        lr_scheduler.step()
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                for val_data in val_loader:
                    val_inputs, val_labels = (
                    val_data["image"].to(device),
                    val_data["mask"].to(device),)
                    val_outputs = inference(val_inputs, model)
                    val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
                    dice_metric(y_pred=val_outputs, y=val_labels)
                    dice_metric_batch(y_pred=val_outputs, y=val_labels)

                metric = dice_metric.aggregate().item()
                metric_values.append(metric)
                metric_batch = dice_metric_batch.aggregate()
                metric_tc = metric_batch[0].item()
                metric_values_tc.append(metric_tc)
                metric_wt = metric_batch[1].item()
                metric_values_wt.append(metric_wt)
                metric_et = metric_batch[2].item()
                metric_values_et.append(metric_et)
                dice_metric.reset()
                dice_metric_batch.reset()

                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    best_metrics_epochs_and_time[0].append(best_metric)
                    best_metrics_epochs_and_time[1].append(best_metric_epoch)
                    best_metrics_epochs_and_time[2].append(time.time() - total_start)
                    torch.save(
                        model.state_dict(),
                        os.path.join(root_dir, "best_metric_model.pth"),)
                    print("saved new best metric model")

                print(
                    f"current epoch: {epoch + 1} current mean dice: {metric:.4f}"
                    f" tc: {metric_tc:.4f} wt: {metric_wt:.4f} et: {metric_et:.4f}"
                    f"\nbest mean dice: {best_metric:.4f}"
                    f" at epoch: {best_metric_epoch}")
        print(f"time consuming of epoch {epoch + 1} is: {(time.time() - epoch_start):.4f}")
    total_time = time.time() - total_start
    return best_metric, best_metric_epoch, total_time, epoch_loss_values, metric_values, metric_values_et, metric_values_tc, metric_values_wt

def plot_results(epoch_loss_values, metric_values,
                 metric_values_et, metric_values_tc,
                 metric_values_wt):
    '''plot training results after training, for example
    loss, mean dice, dice with respect to each output modality'''
    val_interval = 1
    plt.figure("train", (12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Epoch Average Loss")
    x = [i + 1 for i in range(len(epoch_loss_values))]
    y = epoch_loss_values
    plt.xlabel("epoch")
    plt.plot(x, y, color="red")
    plt.subplot(1, 2, 2)
    plt.title("Val Mean Dice")
    x = [val_interval * (i + 1) for i in range(len(metric_values))]
    y = metric_values
    plt.xlabel("epoch")
    plt.plot(x, y, color="green")
    plt.savefig('loss_and_metric_values.png',  bbox_inches='tight')
    plt.show()
   

    plt.figure("train", (18, 6))
    plt.subplot(1, 3, 1)
    plt.title("Val Mean Dice TC")
    x = [val_interval * (i + 1) for i in range(len(metric_values_tc))]
    y = metric_values_tc
    plt.xlabel("epoch")
    plt.plot(x, y, color="blue")
    plt.subplot(1, 3, 2)
    plt.title("Val Mean Dice WT")
    x = [val_interval * (i + 1) for i in range(len(metric_values_wt))]
    y = metric_values_wt
    plt.xlabel("epoch")
    plt.plot(x, y, color="brown")
    plt.subplot(1, 3, 3)
    plt.title("Val Mean Dice ET")
    x = [val_interval * (i + 1) for i in range(len(metric_values_et))]
    y = metric_values_et
    plt.xlabel("epoch")
    plt.plot(x, y, color="purple")
    plt.savefig('output_metrics_values.png',  bbox_inches='tight')
    plt.show()

def evaluate(model, weight, root_dir, val_org_loader, device, post_transforms, 
            dice_metric, dice_metric_batch, metric_batch, metric):
    model.load_state_dict(torch.load(
    os.path.join(root_dir, "best_metric_model.pth")))
    model.eval()
    with torch.no_grad():
        for val_data in val_org_loader:
            val_inputs = val_data["image"].to(device)
            val_data["pred"] = inference(val_inputs)
            val_data = [post_transforms(i) for i in decollate_batch(val_data)]
            val_outputs, val_labels = from_engine(["pred", "mask"])(val_data)
            dice_metric(y_pred=val_outputs, y=val_labels)
            dice_metric_batch(y_pred=val_outputs, y=val_labels)

        metric_org = dice_metric.aggregate().item()
        metric_batch_org = dice_metric_batch.aggregate()

        dice_metric.reset()
        dice_metric_batch.reset()

    metric_tc, metric_wt, metric_et = metric_batch[0].item(), metric_batch[1].item(), metric_batch[2].item()

    print("Metric on original image spacing: ", metric)
    print(f"metric_tc: {metric_tc:.4f}")
    print(f"metric_wt: {metric_wt:.4f}")
    print(f"metric_et: {metric_et:.4f}") 

def pretrained(model, weights):
    """pretrained model to be used for further training
    
    Args:
    model: nn.module 
    weights: str (path) path to pretrained weights  
    
    return:
    model: nn.module --> loaded with pretrained weights"""
    device = torch.device("cuda")
    model.load_state_dict(torch.load(weights, map_location="cuda:0"))
    model.to(device)
    model.train()
    return model

def main():
    args = read_args()
    print('creating root directory... \n')
    root_dir = root_directory(args.data_dir, args.name)
    print('done!!! \n')

    print('-'*60)
    set_determinism(seed=0)
    print('making data loaders to load the Brats20 dataset... \n')
    print('loading training data ... \n')
    survival_df = pd.read_csv(Config.survival_info_csv_train)
    name_mapping_df = pd.read_csv(Config.name_mapping_csv_train)
    df = add_paths(survival_df, name_mapping_df, t= 'train')
    train_dataset, train_loader = load_dataset(df,
                                             'train',
                                              False,
                                              args.workers,
                                              args.batch)
    print('done!!! \n')

    print('-'*60)
    print('loading validation data... \n')
    survival_df_val = pd.read_csv(Config.validation_csv)
    df_val = add_paths(survival_df_val, t="val")
    val_dataset, val_loader = load_dataset(df_val,
                                             'val',
                                              False,
                                              args.workers,
                                              args.batch)
    print('done!!! \n')                         
    print('Loaded successfully!!! \n')
    print('-'*60)
    #--------------------------------------------------------------------------------------
    # print('now loading test dataset \n')
    # survival_df_test = pd.read_csv(Config.survival_info_csv_test)
    # name_mapping_df_test = pd.read_csv(Config.name_mapping_csv_test)
    # df_test = add_paths(survival_df=survival_df_test, name_mapping_df=name_mapping_df_test, t='test')
    # test_dataset, test_loader = load_dataset(df_test,
    #                                          'test',
    #                                          False, 
    #                                          args.workers,
    #                                          args.batch)
    
    # print('Loaded successfully!!! \n')
    #----------------------------------------------------------------------------------------

    device, model, loss_function, optimizer, lr_scheduler, dice_metric, dice_metric_batch, post_trans = model_loss_optim(args.epochs, args.lr, args.weight_decay)
    if args.resume:
        model = pretrained(model, args.weights)
    scaler = torch.cuda.amp.GradScaler()
    # enable cuDNN benchmark
    torch.backends.cudnn.benchmark = True 

    print('Training ... \n')
    print('-'*60)
    best_metric, best_metric_epoch, total_time, epoch_loss, best_metric_values, best_metric_et, best_metric_tc, best_metric_wt = train(args.epochs, model, train_loader, device, optimizer, 
    loss_function, train_dataset, lr_scheduler, val_loader, dice_metric,
    dice_metric_batch, root_dir, scaler, post_trans)
    print('-'*60)
    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}, total time: {total_time}.")
    print('Training Finished!!!!')
    
    plot_results(epoch_loss,best_metric_values, best_metric_et, best_metric_tc, best_metric_wt)
    
    print('Successfully ran everything!!!')

if __name__ == "__main__":
    main()


    
