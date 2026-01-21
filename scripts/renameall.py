import os
import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dir', default='datasets/alpha', type=str, help='Directory where processed images are stored')
parser.add_argument('--gpu', default=0, type=int, help='ID of GPU to use')
args = parser.parse_args()

files = glob.glob(f'{args.dir}/*/0_*_rgba.png')

for file in files:
    print(f'======== processing {file} ========')
    name  = os.path.basename(file)[2:]
    print(name)
    dirname = os.path.dirname(file)
    os.system(f'mv {file} {os.path.join(dirname, name)}')
