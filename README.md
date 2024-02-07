[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/faizan1234567/Brats-20-Tumors-segmentation/blob/main/notebooks/BraTS21_setup.ipynb)
# Brain Tumors Segmentation on BraTS23 Dataset (New)
Brain tumor is one of the deadlist kind of disease around the globe. Among these, gliomas are ubiquitous. Physcians and Radiologists use MRI scans and CT scans to diagnose the disease. In addition, it could be highly prone to human errors, and it's time consuming process when timely diagnosis is required. To assit radioligits, deep learning is being used. Deep Learning has shown remarkable performance in medical imaging over the years. Therefore, to train a deep learning model, a large dataset is being used to train the AI model for accurate diagnosis for early treamtment of the disease.

![Alt Text](https://github.com/faizan1234567/Brats-20-Tumors-segmentation/blob/main/media/gif.gif)

## Dataset Description
All [BraTS23](http://braintumorsegmentation.org/) mpMRI scans are available as [NIfTI](https://radiopaedia.org/articles/nifti-file-format) files and described as  T2 Fluid Attenuated Inversion Recovery(Flair), native(T1), T2-weighted(T2), and post-contrast T1-weighted (T1Gd). They were acquired with differnt clinical protocals and various scanners from different institutions.

Annotations consistsof  GD-enhancing tumor (ET — label 3), the peritumoral edematous/invaded tissue (ED — label 2), and the necrotic tumor core (NCR — label 1), more detail [here](https://www.synapse.org/#!Synapse:syn51156910/wiki/622351).

The dataset contains 1251 patient cases that are labelled by expert radiologists. However, cases in validation set are not labelled. Therefore,the dataset has been divided into five folders. Four of them are used for training and one is used for evaluation.

The dataset is represented in a directory in the following sturcture, please make necessary changes if required in your case. And make sure to add other helping files in the dataset directory.

```
Dataset/
├── training/
│   ├── ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/
│       ├── BraTS-GLI-00000-000/
│       │   ├── BraTS-GLI-00000-000-t2f.nii.gz
│       │   ├── BraTS-GLI-00000-000-seg.nii.gz
│       │   ├── BraTS-GLI-00000-000-t1n.nii.gz
│       │   ├── BraTS-GLI-00000-000-t2w.nii.gz
│       │   └── BraTS-GLI-00000-000-t1c.nii.gz
│       └── ...
│   
└── validation/
    └── ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData/
        ├── BraTS-GLI-00001-000/
        │   ├── BraTS-GLI-00001-000-t1c.nii.gz
        │   ├── BraTS-GLI-00001-000-t2f.nii.gz
        │   ├── BraTS-GLI-00001-000-t1n.nii.gz
        │   └── BraTS-GLI-00001-000-t2w.nii.gz
        └── ...

```

![alt text](https://github.com/faizan1234567/Brats-20-Tumors-segmentation/blob/main/media/fig_brats21.png)
image from  [Baid et al.](https://arxiv.org/pdf/2107.02314v1.pdf)

## Installation

```
git clone https://github.com/faizan1234567/BraTS23-Tumors-Segmentation
cd BraTS23-Tumors-Segmentation
```

create a virtual environment in Anaconda and activate it.
```
conda create -n brats_segmentation python=3.9.0 -y 
conda activate brats_segmentation
```

if you forgot the created enviroment name, you can get it with:
```
conda env list
```

if you mistakenly created an environment with wrong name or it has some issues, you can
remove it and create another one again with a unique name:
```
conda env remove --name your_venv
```

Now install all the dependencies
```
pip install --upgrade pip
pip install -r requirements.txt

# to check packages
pip list
```
installation complete!

## Google Colab Usage
Upload the dataset on your drive. Update the train, json file, and dataset csv file paths in the colab config in the conf directory. Then mount drive with colab:
```python
from google.colab import drive
drive.mount('/gdrive')
```
clone the repo in colab notebook, and then install all the dependencies:
```
!pip install -r requirments.txt
```
and make sure to set colab with command line argument, so the system can load corresponding paths for colab and start loading the data for training.

## Usage
To train on Brats23, run the training command
First configure some paths in configs.py, these are some necessary paths about the dataset directory, data loading, and other important parameters.
```
python train.py -h
optional arguments:
  -h, --help            show this help message and exit
  --data DATA           dataset root directory path
  --fold FOLD           folder name
  --json_file JSON_FILE
                        path to json file
  --batch BATCH         batch size
  --img_roi IMG_ROI     image roi size
  --val_every VAL_EVERY
                        validate every 2 epochs
  --max_epochs MAX_EPOCHS
                        maximum number of epoch to train
  --workers WORKERS     Number of data loading workers
  --pretrained_model PRETRAINED_MODEL
                        path to pretraiend model
  --pretrained          use pretrained weights.
  --resume              starting training from the saved ckpt.
  --platform_changed    pc changed, so that dataset dir has been set accordingly

python train.py --data <dataset-dir> --fold 0 --json_file <dataset-split-file-path>
 --batch 1 --workers 8 --val_every 2 --max_epochs 100 --pretrained_model <pretrained-model-path> 
   ```
To test the model:
```
python test.py -h
optional arguments:
  -h, --help            show this help message and exit
  --weights WEIGHTS     weight file path
  --fold FOLD           fold number for evaluation
  --workers WORKERS     number of workers
  --batch BATCH         batch size to load the dataset
  --json_file JSON_FILE
                        path to the data json file
  --platform_changed    running on other platfrom

python test.py --weights <path-to-weights>  --fold 0 --workers 2 --batch 1 --json_file <dataset-split-file-path>
   ```
To visualize, use:
```
python show.py -h
optional arguments:
  -h, --help            show this help message and exit
  --json_file JSON_FILE
                        dataset split file
  --fold FOLD           folder number
  --phase PHASE         validation or training.
  --save SAVE           results save directory
  --get_abnormal_area   get full abnormal are
  --visualize_data_gif  visulize data gif, and create a gif file
  --visualize_data_sample
                        visualize one sample

python show.py --json_file <path> --fold 0 --phase <"val"> --save <path> 
```

## New Features (to be added)
- MLOps tools integration such as MLflow
- Experiment tracking & Monitoring.
- Multi-GPU training 
- Novel architecture design (private)
- New Data augmentation options (private)
- automatic hyperparameters optimization (private)

## Cite

If you find this project useful in your research, please consider cite:

```latex
@misc{brats23-tumor-segmentation,
    title={Multi-modal BraTS 2023 brain tumor segmentation},
    author={Muhammad Faizan},
    howpublished = {\url{https://github.com/faizan1234567/Brats-20-Tumors-segmentation}},
    year={2023}
}
```

## Acknowledgements
[1]. http://braintumorsegmentation.org/

[2]. https://monai.io/

[3]. https://github.com/Project-MONAI/research-contributions/tree/main/SwinUNETR/BRATS21

[4]. https://github.com/amanchadha/coursera-ai-for-medicine-specialization

[5]. https://arxiv.org/pdf/1906.01796v2.pdf

[6]. https://arxiv.org/abs/2103.14030

[7]. https://arxiv.org/abs/1512.03385