import argparse
import os
from shutil import copyfile

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', default="/home/facit/Faizan/segment/data/MICCAI_BraTS2020_TrainingData-20221206T055905Z-001/MICCAI_BraTS2020_TrainingData",
                        type = str, help= "path to files directory to check files length: ideally there should be 5 files in validatin \
                                           training directories, and 4 files in the test set.")
    parser.add_argument('--src', type=str, help="path to the source dir.")
    opt = parser.parse_args()
    return opt



def check(files_dir):
    """files check in the given dir
    
    Args:
    files_dir: str (path) 
    
    return:
    length: int (number of files in each folder in a dir)"""

    dirs = os.listdir(files_dir)
    i=0
    wrong_files = []
    for dir in dirs:
        # print(dir)
        if "BraTS20_" in dir:
            dir_files = os.path.join(files_dir, dir)
            dir_length = len(os.listdir(dir_files))
            if dir_length == 5:
                continue
            else:
                # print("Error: {}".format(dir))
                i+=1
                wrong_files.append(dir)
    # print('number of wrong dir: {}'.format(i))
    return wrong_files

def correct_files(src, des):
    """correct files in the src dir
    
    Args:
    src: str (path) to the source dir
    des: str (path) to the destination dir"""
    des_dirs = check(des)
    # src_dirs = os.listdirs(src)
    for des_dir in des_dirs:
        print(des_dir)
        if "BraTS20_" in des_dir:
            src_dir_path = os.path.join(src, des_dir)
            des_dir_path = os.path.join(des, des_dir)
            # print(len(os.listdir(src_dir_path)), len(os.listdir(des_dir_path)))
           
            for file in os.listdir(src_dir_path):
                src_file_path = os.path.join(src_dir_path, file)
                des_file_path = os.path.join(des_dir_path, file)
                # if o/s.path.exists(des)
                copyfile(src_file_path, des_file_path)

                




if __name__ == "__main__":
    args = read_args()
    # correct_files(args.src, args.dir)
    check(args.dir)

    print('done!!')
