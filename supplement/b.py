import os
import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dir', default='gso_metric/gs', type=str, help='Directory where processed images are stored')
parser.add_argument('--new_dir', default='gso_metric/gs_new', type=str, help='Directory where processed images are stored')
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
    files = glob.glob(f'{args.dir}/*/0_{int(index*90)}_rgba.png')

    for file in files:
        print(f'======== processing {file} ========')
        dirname = os.path.basename(os.path.dirname(file))
        new_dirname = os.path.join(args.new_dir, dirname)
        os.makedirs(new_dirname, exist_ok=True)
        os.system(f'cp {file} {new_dirname}/{int(index*90)}_rgba.png')
        
for index in range(4):
    cp(index)