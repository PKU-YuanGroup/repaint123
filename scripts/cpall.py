import os
import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dir', default='datasets/alpha', type=str, help='Directory where processed images are stored')
parser.add_argument('--new_dir', default='datasets/alpha', type=str, help='Directory where processed images are stored')
parser.add_argument('--gpu', default=0, type=int, help='ID of GPU to use')
args = parser.parse_args()

files = glob.glob(f'{args.dir}/*/*_rgba.png')

for file in files:
    print(f'======== processing {file} ========')
    dirname = os.path.basename(os.path.dirname(file))
    new_dirname = os.path.join(args.new_dir, dirname)
    os.makedirs(new_dirname, exist_ok=True)
    # cp file to new_dir
    os.system(f'cp {file} {new_dirname}')
