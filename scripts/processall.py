import os
import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dir', default='datasets/alpha', type=str, help='Directory where processed images are stored')
parser.add_argument('--gpu', default=0, type=int, help='ID of GPU to use')
args = parser.parse_args()

files = glob.glob(f'{args.dir}/*/*.png')

for file in files:
    print(f'======== processing {file} ========')
    # # first stage
    # os.system(f'CUDA_VISIBLE_DEVICES={args.gpu} python main.py '
    #           f'--config {configs_dir}/image_realfusion.yaml '
    #           f'input={file} '
    #           f'outdir={out_dir}/{name} '
    #           f'save_path={name} elevation={args.elevation}')
    # second stage
    os.system(f'CUDA_VISIBLE_DEVICES={args.gpu} python process.py '
              f'{file} '
              f'--size 512')
