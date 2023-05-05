'''utility functions'''

import os
import shutil
from configs import Config
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
from   matplotlib import cm
import matplotlib.animation as anim
import matplotlib.patches as mpatches
import imageio
from random import shuffle
import cv2
from IPython.display import Image
# from keras.utils import to_categorical
import tensorflow as tf


def plot_image_grid(image):
    data_all = []

    data_all.append(image)

    fig, ax = plt.subplots(3, 6, figsize=[16, 9])

    # coronal plane
    coronal = np.transpose(data_all, [1, 3, 2, 4, 0])
    coronal = np.rot90(coronal, 1)

    # transversal plane
    transversal = np.transpose(data_all, [2, 1, 3, 4, 0])
    transversal = np.rot90(transversal, 2)

    # sagittal plane
    sagittal = np.transpose(data_all, [2, 3, 1, 4, 0])
    sagittal = np.rot90(sagittal, 1)

    for i in range(6):
        n = np.random.randint(coronal.shape[2])
        ax[0][i].imshow(np.squeeze(coronal[:, :, n, :]))
        ax[0][i].set_xticks([])
        ax[0][i].set_yticks([])
        if i == 0:
            ax[0][i].set_ylabel('Coronal', fontsize=15)

    for i in range(6):
        n = np.random.randint(transversal.shape[2])
        ax[1][i].imshow(np.squeeze(transversal[:, :, n, :]))
        ax[1][i].set_xticks([])
        ax[1][i].set_yticks([])
        if i == 0:
            ax[1][i].set_ylabel('Transversal', fontsize=15)

    for i in range(6):
        n = np.random.randint(sagittal.shape[2])
        ax[2][i].imshow(np.squeeze(sagittal[:, :, n, :]))
        ax[2][i].set_xticks([])
        ax[2][i].set_yticks([])
        if i == 0:
            ax[2][i].set_ylabel('Sagittal', fontsize=15)

    fig.subplots_adjust(wspace=0, hspace=0)


def get_labeled_image(image, label, is_categorical=False):

        if not is_categorical:
            # print(image.shape, label.shape)
            label = tf.keras.utils.to_categorical(label, num_classes=4)#.astype(np.uint8)

        image = cv2.normalize(image[:, :, :, 0], None, alpha=0, beta=255,
                            norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F).astype(
            np.uint8)

        labeled_image = np.zeros_like(label[:, :, :, 1:])

        # remove tumor part from image
        labeled_image[:, :, :, 0] = image * (label[:, :, :, 0])
        labeled_image[:, :, :, 1] = image * (label[:, :, :, 0])
        labeled_image[:, :, :, 2] = image * (label[:, :, :, 0])

        # color labels
        labeled_image += label[:, :, :, 1:] * 255
        return labeled_image



class util:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.LGG18 = Config.LGG18
        self.HGG18 = Config.HGG18
        self.brats19 = Config.brats19_test_data
        # self.process_mask = process_mas

    def move_data(self):
        '''move some files from HGG folder and Some files from LGG
        folder for creating a test set
        LGG: path
        HGG: path'''
        LGG_files = os.listdir(self.LGG18)
        HGG_files = os.listdir(self.HGG18)
        dest_dir = self.brats19
        files_to_copy = 20
        #merge files
        files = HGG_files + LGG_files
        shuffle(files)
        for file in files[:2*files_to_copy]:
            hgg_file = os.path.join(self.HGG18, file)
            if "HGG" in hgg_file:
                dest_file = os.path.join(dest_dir, file)
                if os.path.exists(hgg_file):
                    if len(os.listdir(hgg_file)) == 5:
                        shutil.copytree(hgg_file, dest_file)
                    else:
                        print(f'{file} patient data is incomplete!!!')
                else:
                    print(f' {hgg_file} path doesnt exists')
            else:
                lgg_file = os.path.join(self.LGG18, file)
                dest_file = os.path.join(dest_dir, file)
                if os.path.exists(lgg_file):
                    if len(os.listdir(lgg_file)) == 5:
                        shutil.copytree(lgg_file, dest_file)
                    else: 
                        print(f'{file} patient data is incomplete!!!')
                else:
                    print(f'{lgg_file} path doesnt exists')
        print('Copied successfully!!!')

    @staticmethod
    def move_directories(source, destination):
        '''move directories from soruce to desintation
        source: path (str), LGG or HGG directory
        destination: path (str), test set directory'''
        if 'HGG' in source:   
            files = os.listdir(source)[:Config.Testset.HGG_files]
            for dir in files:
                src_path = os.path.join(source, dir)
                dest_path = os.path.join(destination, dir)
                shutil.move(src_path, dest_path)
        else:
            files = os.listdir(source)[:Config.Testset.LGG_files]
            for dir in files:
                src_path = os.path.join(source, dir)
                dest_path = os.path.join(destination, dir)
                shutil.move(src_path, dest_path)


    @staticmethod
    def inspect_data(patient_data):
        '''inspect input images and labels
        patient_data: path (str)
        path to MRI modalitiets files and label'''
        files = os.listdir(patient_data)
        modalities = []
        for modality in files:
            if not '_seg.nii.gz' in modality:
                modality_path = os.path.join(patient_data, modality)
                img = nib.load(modality_path)
                img = np.asanyarray(img.dataobj)
                img = np.rot90(img)
                modalities.append(img)
        id = patient_data.split("/")[-1]
        print('patient id: {}'.format(id))
        label_path = os.path.join(patient_data, id+ '_seg.nii.gz')
        label = nib.load(label_path)
        label = np.asanyarray(label.dataobj)
        label = np.rot90(label)
        print('shape of the label: {}'.format(label.shape))
        mask = util.process_mask(label)
        util.visualize(modalities, mask)



    @staticmethod
    def process_mask(sample_mask):
        mask_WT = sample_mask.copy()
        mask_WT[mask_WT == 1] = 1
        mask_WT[mask_WT == 2] = 1
        mask_WT[mask_WT == 4] = 1

        mask_TC = sample_mask.copy()
        mask_TC[mask_TC == 1] = 1
        mask_TC[mask_TC == 2] = 0
        mask_TC[mask_TC == 4] = 1

        mask_ET = sample_mask.copy()
        mask_ET[mask_ET == 1] = 0
        mask_ET[mask_ET == 2] = 0
        mask_ET[mask_ET == 4] = 1
        return (mask_WT, mask_TC, mask_ET)
    
    @staticmethod
    def visualize(modalities, mask):
        #unpack mask tuple
        mask_WT, mask_TC, mask_ET = mask
        fig = plt.figure(figsize= (20, 10))
        gs = gridspec.GridSpec(nrows=2, ncols=4, height_ratios = [1, 1.5])
        #  Varying density along a streamline
        modality_names = ['flair', 't1', 't1ce', 't2']
        axis = []
        for j, modality in enumerate(modalities):
            ax = fig.add_subplot(gs[0, j])
            modality_img = ax.imshow(modality[:,:,65], cmap='bone')
            ax.set_title(f"{modality_names[j]}", fontsize=18, weight='bold', y=-0.2)
            fig.colorbar(modality_img)
            axis.append(ax)
        #  Varying density along a streamline
        ax4 = fig.add_subplot(gs[1, 1:3])

        #ax4.imshow(np.ma.masked_where(mask_WT[:,:,65]== False,  mask_WT[:,:,65]), cmap='summer', alpha=0.6)
        l1 = ax4.imshow(mask_WT[:,:,65], cmap='summer',)
        l2 = ax4.imshow(np.ma.masked_where(mask_TC[:,:,65]== False,  mask_TC[:,:,65]), cmap='rainbow', alpha=0.6)
        l3 = ax4.imshow(np.ma.masked_where(mask_ET[:,:,65] == False, mask_ET[:,:,65]), cmap='winter', alpha=0.6)

        ax4.set_title("", fontsize=20, weight='bold', y=-0.1)
        axis.append(ax4)
        _ = [ax.set_axis_off() for ax in axis]

        colors = [im.cmap(im.norm(1)) for im in [l1,l2, l3]]
        labels = ['Non-Enhancing tumor core', 'Peritumoral Edema ', 'GD-enhancing tumor']
        patches = [mpatches.Patch(color=colors[i], label=f"{labels[i]}") for i in range(len(labels))]
        # put those patched as legend-handles into the legend
        plt.legend(handles=patches, bbox_to_anchor=(1.1, 0.65), loc=2, borderaxespad=0.4,fontsize = 'xx-large',
                title='Mask Labels', title_fontsize=18, edgecolor="black",  facecolor='#c5c6c7')

        plt.suptitle("Multimodal Scans  | Manually-Annontated mask - Target", fontsize=20, weight='bold')

        fig.savefig("data_sample.png", format="png",  pad_inches=0.2, transparent=False, bbox_inches='tight')
        fig.savefig("data_sample.svg", format="svg",  pad_inches=0.2, transparent=False, bbox_inches='tight')
    
    @staticmethod
    def visualize_data_gif(data_):
        images = []
        for i in range(data_.shape[0]):
            x = data_[min(i, data_.shape[0] - 1), :, :]
            y = data_[:, min(i, data_.shape[1] - 1), :]
            z = data_[:, :, min(i, data_.shape[2] - 1)]
            img = np.concatenate((x, y, z), axis=1)
            images.append(img)
        imageio.mimsave("/tmp/gif.gif", images, duration=0.01)
        return Image(filename="/tmp/gif.gif", format='png')

    @staticmethod   
    def load_case(image_nifty_file, label_nifty_file):
        # load the image and label file, get the image content and return a numpy array for each
        image = np.array(nib.load(image_nifty_file).get_fdata())
        label = np.array(nib.load(label_nifty_file).get_fdata())
        
        return image, label  
          
    @staticmethod
    def separate_label(case):
        '''separate an MRI image to it's label
        case: patient id with mri modaliteis (path)
        return
        image, label'''
        # global mask
        if os.path.exists(case):
            image = []
            label = []
            contents = os.listdir(case)
            for modality in contents:
                if not '_seg.nii.gz' in modality:
                    file_path = os.path.join(case, modality)
                    data = nib.load(file_path)
                    data = np.asarray(data.dataobj)
                    data_min = np.min(data)
                    img = (data - data_min) / (np.max(data) - data_min)
                    image.append(img)
                else:

                    mask_path =  os.path.join(case, modality)
                    mask = nib.load(mask_path)
                    mask = np.asanyarray(mask.dataobj)
                    # mask_WT = mask.copy()
                    # mask_WT[mask_WT == 1] = 1
                    # mask_WT[mask_WT == 2] = 1
                    # mask_WT[mask_WT == 4] = 1

                    # mask_TC = mask.copy()
                    # mask_TC[mask_TC == 1] = 1
                    # mask_TC[mask_TC == 2] = 0
                    # mask_TC[mask_TC == 4] = 1

                    # mask_ET = mask.copy()
                    # mask_ET[mask_ET == 1] = 0
                    # mask_ET[mask_ET == 2] = 0
                    # mask_ET[mask_ET == 4] = 1
                     
                    # mask = np.stack([mask_WT, mask_TC, mask_ET])
                    mask = np.moveaxis(mask, (0, 1, 2), (1, 2, 0))
                    label.append(mask)

            global image_data       
            image_data = np.stack(image)
            image_data = np.moveaxis(image_data, (0, 1, 2, 3), (1, 2, 3, 0))
            return (image_data, label[0])

                    

                        






                


