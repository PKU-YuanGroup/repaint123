import os
import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dir', default='supplement/syncdreamer', type=str, help='Directory where processed images are stored')
parser.add_argument('--new_dir', default='supplement/syncdreamer_new', type=str, help='Directory where processed images are stored')
parser.add_argument('--gpu', default=7, type=int, help='ID of GPU to use')
args = parser.parse_args()
 
def index2i(index):
    if index == 0:
        return 0
    elif index == 1:
        return 3
    elif index == 2:
        return 2
    elif index == 3:
        return 1   
    
def cp(index):
    files = glob.glob(f'{args.dir}/*/*000{index}*.png')

    for file in files:
        print(f'======== processing {file} ========')
        dirname = os.path.basename(os.path.dirname(file))
        new_dirname = os.path.join(args.new_dir, dirname)
        os.makedirs(new_dirname, exist_ok=True)
        # cp file to new_dir
        os.system(f'cp {file} {new_dirname}')
        os.system(f'mv {new_dirname}/{os.path.basename(file)} {new_dirname}/{index}.png')
        # os.system(f'mv {new_dirname}/{os.path.basename(file)} {new_dirname}/{index2i(index)}.png')
        
for index in range(4):
    cp(index)