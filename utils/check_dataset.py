"""
==============================================================================
check the brats23 dataset if it has all the files for validation purposes
if it miss a modality or any label then it will create problem during training
===============================================================================


python check_dataset.py
"""
import os
import argparse 




def check(directory: str, type = 'train'):
    ''''
    check number of folder in directory if they match with the
    total patients cases in the training data

    then for each directory check if each of 5 modalities given
    T1, T1ce, T2, FLAIR, and seg label

    Parameters
    ----------
    directory: training or validation directory
    type: if train 5, if validatoin or test it could be less than 5
    '''
    cases = os.listdir(directory)
    num_cases = len(cases)

    print(f'There are {num_cases} cases')

    # check each folder should have 5 mod
    for case in cases:
        case_path = os.path.join(directory, case)
        modalities = os.listdir(case_path)
        num_modalities = len(modalities)
        if type == 'train':
            assert num_modalities == 5, f"Error: There should be 5 modalities in {case} but found {num_modalities}"
        else:
            assert num_modalities == 4, f"Error: There should be 4 modalities in {case} but found {num_modalities}"
    print('INFO: No issue found')
    
        
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type = str, default= "E:/Brats21 Data/Dataset/training/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/",
                        help = 'path to one of data directory either training or val')
    parser.add_argument('--type', type = str, default='train', help= 'train or val data')
    args = parser.parse_args()

    check(args.dir, args.type)



