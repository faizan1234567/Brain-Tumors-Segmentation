'''set up the config for brats dataset, train and test folders names and
some parameters'''
import os
import torch
import monai
from monai.metrics import DiceMetric
from monai.utils.enums import MetricReduction
from monai.networks.nets import SwinUNETR
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai import transforms
from monai.transforms import (
    AsDiscrete,
    Activations,
)
from functools import partial

class Config:
    root_dir = "/gdrive/MyDrive/THESIS BraTs NUST"
    train_root_dir = "/gdrive/MyDrive/THESIS BraTs NUST/MICCAI_BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"
    test_root_dir  = "/gdrive/MyDrive/THESIS BraTs NUST/MICCAI_BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData"
    path_to_train_csv = "/gdrive/MyDrive/THESIS BraTs NUST/train_data.csv"
    seed = 50
    survival_info_csv_train = "/gdrive/MyDrive/THESIS BraTs NUST/MICCAI_BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/survival_info.csv"
    name_mapping_csv_train = "/gdrive/MyDrive/THESIS BraTs NUST/MICCAI_BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/name_mapping.csv"
    survival_info_csv_test = "/gdrive/MyDrive/THESIS BraTs NUST/MICCAI_BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData/survival_evaluation.csv"
    name_mapping_csv_test = "/gdrive/MyDrive/THESIS BraTs NUST/MICCAI_BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData/name_mapping_validation_data.csv"
    validation_csv = "/gdrive/MyDrive/THESIS BraTs NUST/MICCAI_BraTSXX_TestData/Testset/test_set.csv"
    a_test_patient = "/gdrive/MyDrive/THESIS BraTs NUST/MICCAI_BraTSXX_TestData/Testset/BraTS19_TCIA08_167_1"
    LGG18 = "/gdrive/MyDrive/Brats18/LGG"
    HGG18 = "/gdrive/MyDrive/Brats18/HGG"
    data_dir = "/gdrive/MyDrive/THESIS BraTs NUST"
    brats18_val_data = "/gdrive/MyDrive/THESIS BraTs NUST/MICCAI_BraTSXX_TestData/Testset"
    brats19_test_data = "/gdrive/MyDrive/THESIS BraTs NUST/MICCAI_BraTS19_Test/MICCAI_BraTS19_Test"
    brats19_test_survival_csv = "/gdrive/MyDrive/THESIS BraTs NUST/MICCAI_BraTS19_Test/MICCAI_BraTS19_Test/test_set.csv"
    
    class Testset:
        Brat19_HGG_path = "/gdrive/MyDrive/Medical Dataset/MICCAI_BraTS_2019_Data_Training/HGG"
        Brats19_LGG_path = "/gdrive/MyDrive/Medical Dataset/MICCAI_BraTS_2019_Data_Training/LGG"
        Brats19_TestData = "/gdrive/MyDrive/Medical Dataset/MICCAI_BraTS_2019_Data_Training/Testset"
        size = 41
        HGG_files = 21
        LGG_files = 20
        val_dir_path = "/gdrive/MyDrive/THESIS BraTs NUST/MICCAI_BraTSXX_TestData/Testset"
        # '/gdrive/MyDrive/THESIS BraTs NUST/MICCAI_BraTSXX_TestData/Testset'
    class newGlobalConfigs:
        root_dir = "E:/Brats21 Data/training"
        train_root_dir = root_dir + "/RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021"+ "/RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021"
        test_root_dir = "E:/Brats21 Data/validation/RSNA_ASNR_MICCAI_BraTS2021_ValidationData/RSNA_ASNR_MICCAI_BraTS2021_ValidationData"
        path_to_csv = root_dir + "/BraTS21-17_Mapping.csv"
        pretrained_model = ""
        survival_info_df = ""
        a_test_patient = "BraTS2021_00000" # change it to other patients, if needed
        full_patient_path = train_root_dir +"/" + a_test_patient
        name_mapping_df = path_to_csv
        seed = 50
        class swinUNetCongis:
            roi = (128, 128, 128)
            batch_size = 2
            fold = 1
            max_epochs = 100
            infer_overlap = 0.5
            val_every = 10
            class training_cofigs:
                roi = (128, 128, 128)
                sw_batch_size = 4
                infer_overlap = 0.5
                dice_loss = DiceLoss(to_onehot_y=False, sigmoid=True)
                post_simgoid = Activations(sigmoid= True)
                post_pred = AsDiscrete(argmax= False, threshold= 0.5)
                dice_acc = DiceMetric(include_background=True, reduction=MetricReduction.MEAN_BATCH, 
                                      get_not_nans=True)
                device = "cuda" if torch.cuda.is_available() else "cpu"
                model = SwinUNETR(
                                    img_size=roi,
                                    in_channels=4,
                                    out_channels=3,
                                    feature_size=48,
                                    drop_rate=0.0,
                                    attn_drop_rate=0.0,
                                    dropout_path_rate=0.0,
                                    use_checkpoint=True,
                                ).to(device)
                
                model_inferer = partial(
                                        sliding_window_inference,
                                        roi_size=[roi[0], roi[1], roi[2]],
                                        sw_batch_size=sw_batch_size,
                                        predictor=model,
                                        overlap=infer_overlap)


