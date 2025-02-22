"""
========================================================
Train Multi-Modal BraTS dataset Tumor segmentation model
on the BraTS dataset
========================================================

Author: Muhammad Faizan

All right reserved
=========================================================
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import time
import gc
import nibabel as nib
import tqdm as tqdm
from utils.meter import AverageMeter
from utils.general import save_checkpoint, load_pretrained_model, resume_training
from brats import get_datasets

from monai.data import  decollate_batch
import torch
import torch.nn as nn
from torch.backends import cudnn

from monai.metrics import DiceMetric
from monai.utils.enums import MetricReduction
from networks.models.ResUNetpp.model import ResUnetPlusPlus
from monai.losses import DiceLoss, DiceCELoss
from monai.inferers import sliding_window_inference
from monai.transforms import (
    AsDiscrete,
    Activations,
)
from monai.networks.nets import SwinUNETR, SegResNet, VNet, AttentionUnet, UNETR
from networks.models.ResUNetpp.model import ResUnetPlusPlus
from networks.models.UNet.model import UNet3D
from networks.models.UX_Net.network_backbone import UXNET
from networks.models.nnformer.nnFormer_tumor import nnFormer
try:
    from thesis.models.SegUXNet.model import SegUXNet
except ModuleNotFoundError:
    print('model not available, please train with other models')
    
from functools import partial
from utils.augment import DataAugmenter
from utils.schedulers import SegResNetScheduler, PolyDecayScheduler

# Configure logger
import logging
import hydra
from omegaconf import DictConfig

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

os.makedirs("logger", exist_ok=True)
file_handler = logging.FileHandler(filename= "logger/train_logger.log")
stream_handler = logging.StreamHandler()
formatter = logging.Formatter(fmt= "%(asctime)s: %(message)s", datefmt= '%Y-%m-%d %H:%M:%S')
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)

class CIdentity(nn.Module):
    """identify mapping a list of values """
    def __init__(self):
        super().__init__()
        self.model = nn.Identity()
    
    def forward(self, x, y):
        x = nn.Identity(x)
        y = nn.Identity(y)
        return (x, y)
    
class Solver:
    """list of optimizers for training NN"""
    def __init__(self, model: nn.Module, lr: float = 1e-4, weight_decay: float = 1e-5):
        self.lr = lr
        self.weight_decay = weight_decay

        self.all_solvers = {
            "Adam": torch.optim.Adam(model.parameters(), lr=self.lr, 
                                     weight_decay= self.weight_decay, 
                                     amsgrad=True), 
            "AdamW": torch.optim.AdamW(model.parameters(), lr=self.lr, 
                                     weight_decay= self.weight_decay, 
                                     amsgrad=True),
            "SGD": torch.optim.SGD(model.parameters(), lr=self.lr, 
                                     weight_decay= self.weight_decay),
        }
    def select_solver(self, name):
        return self.all_solvers[name]
    

def save_best_model(dir_name, model, name="best_model"):
    """save best model weights"""
    save_path = os.path.join(dir_name, name)
    torch.save(model.state_dict(), f"{save_path}/{name}.pkl")
    
def save_checkpoint(dir_name, state, name="checkpoint"):
    """save checkpoint with each epoch to resume"""
    save_path = os.path.join(dir_name, name)
    torch.save(state, f"{save_path}/{name}.pth.tar")
 
def compute_loss(loss, preds, label):
        loss = loss(preds[0], label)
        for i, pred in enumerate(preds[1:]):
            downsampled_label = nn.functional.interpolate(label, pred.shape[2:])
            loss += 0.5 ** (i + 1) * loss(pred, downsampled_label)
        c_norm = 1 / (2 - 2 ** (-len(preds)))
        return c_norm * loss


def create_dirs(dir_name):
    """create experiment directory storing
    checkpoint and best weights"""
    os.makedirs(dir_name, exist_ok=True)
    os.makedirs(os.path.join(dir_name, "checkpoint"), exist_ok=True)
    os.makedirs(os.path.join(dir_name, "best-model"), exist_ok=True)

def init_random(seed):
    """randomly initialize some options"""
    torch.manual_seed(seed)        
    torch.cuda.manual_seed(seed)  
    torch.cuda.manual_seed_all(seed) 
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    cudnn.benchmark = False         
    cudnn.deterministic = True

# Train for an epoch
def train_epoch(model, loader, optimizer, loss_func, augment = True):
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
    # dyn_loss = LossBraTS(focal=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    augmenter = DataAugmenter().to(device)
    torch.cuda.empty_cache()
    gc.collect()
    # del variables
    model.train() 
    run_loss = AverageMeter()
    for batch_data in loader:
        image, label = batch_data["image"].to(device), batch_data["label"].to(device)
        image, label = augmenter(image, label) if augment else (image, label)
        logits = model(image)
        loss = loss_func(logits, label) 
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        run_loss.update(loss.item(), n = batch_data["image"].shape[0])
    torch.cuda.empty_cache()
    return run_loss.avg

# Validate the model
def val(model, loader, acc_func, model_inferer = None,
        post_sigmoid = None, post_pred = None, post_label=None):
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
    run_acc = AverageMeter()
    with torch.no_grad():
        for batch_data in loader:
            logits = model_inferer(batch_data["image"].to(device))
            masks = decollate_batch(batch_data["label"].to(device)) 
            prediction_lists = decollate_batch(logits)
            predictions = [post_pred(post_sigmoid(prediction)) for prediction in prediction_lists]
            acc_func.reset()
            acc_func(y_pred = predictions, y = masks)
            acc, not_nans = acc_func.aggregate()
            run_acc.update(acc.cpu().numpy(), n = not_nans.cpu().numpy())
    return run_acc.avg

# Save trained results
def save_data(training_loss,
              et, wt, tc,
              val_mean_acc,
              epochs, cfg):
    """
    save the training data for later use
    
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
    save_path = os.path.join(cfg.training.exp_name, "csv")
    os.makedirs(save_path, exist_ok= True)
    data_df.to_csv(os.path.join(save_path, "training_data.csv"))
    return data

def trainer(cfg,
            model,
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
            post_label = None,
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
        epoch_time = time.time()
        training_loss = train_epoch(model = model,
                                    loader = train_loader,
                                    optimizer = optimizer,
                                    loss_func = loss_func)
        print(
            "Epoch  {}/{},".format(epoch + 1, max_epochs),
            "loss: {:.4f},".format(training_loss),
            "time {:.2f} m,".format((time.time() - epoch_time)/60),
            end=""
        )

        if epoch % val_every == 0 or epoch == 0:
            epoch_losses.append(training_loss)
            train_epochs.append(int(epoch))
            val_acc =  val(model = model,
                          loader = val_loader,
                          acc_func = acc_func,
                          model_inferer= model_inferer,
                          post_sigmoid=post_sigmoid,
                          post_pred=post_pred, 
                          post_label = post_label)
            dice_tc = val_acc[0]
            dice_wt = val_acc[1]
            dice_et = val_acc[2]
            val_mean_acc = np.mean(val_acc)
            print(
                " Validation: "
                "dice_tc:", "{:.4f},".format(dice_tc),
                " dice_wt:", "{:.4f},".format(dice_wt),
                " dice_et:", "{:.4f},".format(dice_et),
                " mean_dice:", "{:.4f}".format(val_mean_acc))
            
            dices_tc.append(dice_tc)
            dices_et.append(dice_et)
            dices_wt.append(dices_wt)
            mean_dices.append(val_mean_acc)
            if val_mean_acc > val_acc_max:
                val_acc_max = val_mean_acc
                save_best_model(cfg.training.exp_name, model, "best-model")
            scheduler.step()
            save_checkpoint(cfg.training.exp_name, dict(epoch=epoch + 1, max_epochs=max_epochs, model = model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict()), "checkpoint")
    print()
    print("Training Finished!, Best Accuracy: ", val_acc_max)

    # Save important data
    save_data(training_loss=training_loss,
              et= dices_et,
              wt= dices_wt,
              tc=dices_tc,
              val_mean_acc=mean_dices,
              epochs=train_epochs, 
              cfg = cfg)
    
    return (
        val_acc_max,
        dices_tc,
        dices_wt,
        dices_et,
        mean_dices,
        training_loss,
        train_epochs)

def run(cfg, model,
        loss_func,
        acc_func,
        optimizer,
        train_loader,
        val_loader,
        scheduler,
        model_inferer = None,
        post_sigmoid = None, 
        post_pred = None,
        post_label = None,
        max_epochs = 100,
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
    # Create experiments folders
    create_dirs(cfg.training.exp_name)

    # resume 
    if cfg.training.resume:
        print('Resuming training...')
        checkpoint = torch.load(os.path.join(cfg.training.exp_name, "checkpoint", "checkpoint.pth.tar"))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        # Extend training from saved ckpt
        if cfg.training.new_max_epochs is not None:
            max_epochs = cfg.training.new_max_epochs
        else:
            max_epochs = checkpoint["max_epochs"]
        scheduler.load_state_dict(checkpoint['scheduler'])
        print(f"start train from epoch = {start_epoch}/{max_epochs}")

    else:
        # Training from scratch
        print('Trainig from scrath!')
        start_epoch = 0

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
        cfg, 
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_func=loss_func,
        acc_func=acc_func,
        scheduler=scheduler,
        model_inferer=model_inferer,
        start_epoch=start_epoch,
        max_epochs= max_epochs,
        post_sigmoid=post_sigmoid,
        val_every=val_every,
        post_pred=post_pred,
        post_label=post_label
    )
    print()
    return (val_mean_dice_max, 
            dices_tc,
            dices_wt,
            dices_et,
            dices_mean,
            train_losses,
            train_epochs)

@hydra.main(config_name='configs', config_path= 'conf', version_base=None)
def main(cfg: DictConfig):

    # Initialize random
    init_random(seed=cfg.training.seed)

    # CUDA or CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Efficient training
    torch.backends.cudnn.benchmark = True

    # BraTS configs
    if cfg.dataset.type == "brats":
        num_classes = 3
        in_channels = 4
        crop_size = (128, 128, 128)
        post_pred = AsDiscrete(argmax=False, threshold=0.5)
        post_sigmoid = Activations(sigmoid=True)
        
    spatial_size = 3


    # SegResNet
    if cfg.model.architecture == "segres_net":
        model = SegResNet(spatial_dims=spatial_size, 
                          init_filters=32, 
                          in_channels=in_channels, 
                          out_channels=num_classes, 
                          dropout_prob=0.2, 
                          blocks_down=(1, 2, 2, 4), 
                          blocks_up=(1, 1, 1)).to(device)
    # UNET
    elif cfg.model.architecture == "unet3d":
        model = UNet3D(in_channels=in_channels, 
                       num_classes=num_classes).to(device)
        
    # VNet
    elif cfg.model.architecture == "v_net":
        model = VNet(spatial_dims=spatial_size, 
                     in_channels=in_channels, 
                     out_channels=num_classes,
                     dropout_dim=1,
                     bias= False
                        ).to(device)
    # Attention UNet
    elif cfg.model.architecture == "attention_unet":
        model = AttentionUnet(spatial_dims=spatial_size, 
                              in_channels=in_channels, 
                              out_channels=num_classes, 
                              channels= (8, 16, 32, 64, 128), 
                              strides = (2, 2, 2, 2),
                                           ).to(device)
    # ResUNetpp
    elif cfg.model.architecture == "resunet_pp":
        model = ResUnetPlusPlus(in_channels=in_channels,
                                out_channels=num_classes).to(device)
    # UNETR
    elif cfg.model.architecture == "unet_r":
       model =  UNETR(in_channels=in_channels, 
                     out_channels=num_classes, 
                     img_size=crop_size, 
                     proj_type='conv', 
                     norm_name='instance').to(device)
    # SwinUNETR
    elif cfg.model.architecture == "swinunet_r":
        model = SwinUNETR(
                img_size=crop_size[0],
                in_channels=in_channels,
                out_channels=num_classes,
                feature_size=48,
                drop_rate=0.1,
                attn_drop_rate=0.2,
                dropout_path_rate=0.1,
                spatial_dims=spatial_size,
                use_checkpoint=False,
                use_v2=False).to(device)
    # UXNet
    elif cfg.model.architecture == "ux_net":
        model = UXNET(in_chans= in_channels, 
                      out_chans= num_classes,
                      depths=[2, 2, 2, 2],
                      feat_size=[48, 96, 192, 384],
                      drop_path_rate=0,
                      layer_scale_init_value=1e-6, 
                      spatial_dims=spatial_size).to(device)
    
    # nnFormer
    elif cfg.model.architecture == "nn_former":
        model = nnFormer(crop_size=np.array(crop_size), 
                         embedding_dim=96, 
                         input_channels=in_channels, 
                         num_classes=num_classes, 
                         depths=[2, 2, 2, 2], 
                         num_heads=[3, 6, 12, 24], 
                         deep_supervision=False,
                         conv_op=nn.Conv3d,
                         patch_size= [4,4,4], 
                         window_size=[4,4,8,4]).to(device)
    # SegConvNet
    elif cfg.model.architecture == "seg_uxnet":
        model = SegUXNet(spatial_dims=3, 
                         init_filters=32, 
                         in_channels= in_channels,
                         out_channels=num_classes, 
                         dropout_prob=0.2, 
                         blocks_down=(1, 2, 2, 4), 
                         blocks_up=(1, 1, 1), 
                         enable_gc=True).to(device)
        
    print('Chosen Network Architecture: {}'.format(cfg.model.architecture))
    roi = cfg.model.roi

    # Sliding window inference on evaluation dataset.
    model_inferer = partial(
                        sliding_window_inference,
                        roi_size=[roi] * 3, 
                        sw_batch_size=cfg.training.sw_batch_size,
                        predictor=model,
                        overlap=cfg.model.infer_overlap)
    
    # Validation frequency.
    val_every = cfg.training.val_every

    # Dice or Dice and Cross Entropy loss combined
    if cfg.training.loss_type == "dice" and cfg.dataset.type == 'brats':
        loss_func = DiceLoss(to_onehot_y=False, sigmoid=True)
    elif cfg.training.loss_type == "dice" and cfg.dataset.type == 'btcv':
        loss_func =  DiceCELoss(to_onehot_y=True, softmax=True)
    elif cfg.training.loss_type == "dice_ce":
        loss_func = DiceCELoss(to_onehot_y=False, sigmoid=True)

    # Dice metric 
    if cfg.dataset.type == 'brats':
        acc_func =  DiceMetric(include_background=True, reduction=MetricReduction.MEAN_BATCH, 
                                        get_not_nans=True)
    # Optimizer
    solver = Solver(model=model, lr=cfg.training.learning_rate, 
                       weight_decay=cfg.training.weight_decay)
    optimizer = solver.select_solver(cfg.training.solver_name)

     # Max epochs
    max_epochs = cfg.training.max_epochs

    # Learning rate schedulers
    if cfg.model.architecture == "segres_net":
        scheduler = SegResNetScheduler(optimizer, max_epochs, cfg.training.learning_rate)
    elif cfg.model.architecture == "nn_former":
        scheduler = PolyDecayScheduler(optimizer, total_epochs=max_epochs, initial_lr=cfg.training.learning_rate)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    
    # Batch and worker 
    batch_size = cfg.training.batch_size
    num_workers = cfg.training.num_workers
    
    # Set path to dataset (Customize to your case in configs)
    if cfg.training.colab:
        dataset_dir = cfg.dataset.colab
    elif cfg.training.irl:
        dataset_dir = cfg.dataset.irl_pc
    elif cfg.training.sines:
        dataset_dir = cfg.dataset.sines_pc
    elif cfg.training.my_pc:
        dataset_dir = cfg.dataset.laptop_pc


    train_dataset = get_datasets(dataset_dir, "train", target_size=(128, 128, 128))
    train_val_dataset = get_datasets(dataset_dir, "train_val", target_size=(128, 128, 128))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, 
                                            shuffle=True, num_workers=num_workers, 
                                            drop_last=False, pin_memory=True)
    
    val_loader = torch.utils.data.DataLoader(train_val_dataset, 
                                            batch_size=batch_size, 
                                            shuffle=False, num_workers=num_workers, 
                                            pin_memory=True)
    # Training
    run(cfg, model=model,
        loss_func= loss_func,
        acc_func= acc_func,
        optimizer= optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        scheduler=scheduler,
        model_inferer=model_inferer,
        post_label = None,
        post_sigmoid = post_sigmoid,
        post_pred=post_pred,
        max_epochs=max_epochs,
        val_every=val_every)
        
if __name__ == "__main__":
    main()