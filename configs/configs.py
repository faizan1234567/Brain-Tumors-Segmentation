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
    class newGlobalConfigs:
        root_dir = "E:/Brats21 Data/Dataset"
        train_root_dir = root_dir + "/training/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
        test_root_dir = root_dir + "/validation/ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData"
        path_to_xlsx = root_dir + "/BraTS2023_2017_GLI_Mapping.xlsx"
        pretrained_model = ""
        survival_info_df = ""
        a_test_patient = "BraTS-GLI-00016-000" # change it to other patients, if needed
        full_patient_path = train_root_dir +"/" + a_test_patient
        name_mapping_df = path_to_xlsx
        seed = 50
        json_file = root_dir + "/dataset.json"
        
        class OtherPC:
            root_dir = "/content"
            train_root_dir = root_dir + "/RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021"
            path_to_csv = root_dir + "/BraTS21-17_Mapping.csv"
            a_test_patient = "BraTS2021_00000" # change it to other patients, if needed
            full_patient_path = train_root_dir +"/" + a_test_patient
            name_mapping_df = path_to_csv
            seed = 50
            json_file = root_dir + "/brats23_folds.json"

        class swinUNetCongis:
            roi = (128, 128, 128)
            fold = 1
            max_epochs = 100
            infer_overlap = 0.5
            val_every = 2
            class training_cofigs:
                roi = (128, 128, 128)
                num_workers = 2
                batch_size = 1
                sw_batch_size = 1
                infer_overlap = 0.6
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
                learning_rate = 1e-4
                weight_decay = 1e-5
                max_epochs = 100
                optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, 
                                              weight_decay=weight_decay)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)


