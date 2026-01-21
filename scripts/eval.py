import os
import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--file', default='/data/aigc/tzy/zjw/dreamgaussian/data2/bird_2_rgba.png', type=str, help='Directory where processed images are stored')
parser.add_argument('--obj', default='logs_realfusion/bird_2/bird_2.obj', type=str, help='Directory where obj files will be saved')
parser.add_argument('--gpu', default=7, type=int, help='ID of GPU to use')
args = parser.parse_args()

os.system(f'CUDA_VISIBLE_DEVICES={args.gpu} python clip.py '
            f'{args.file} '
            f'{args.obj} '
            '--force_cuda_rast')
# psnr
os.system(f'CUDA_VISIBLE_DEVICES={args.gpu} python psnr.py '
            f'{args.file} '
            f'{args.obj} '
            '--force_cuda_rast')
# contextial
os.system(f'CUDA_VISIBLE_DEVICES={args.gpu} python contextual.py '
            f'{args.file} '
            f'{args.obj} '
            '--force_cuda_rast')
# # fid
# os.system(f'CUDA_VISIBLE_DEVICES={args.gpu} python fid.py '
#             f'{args.file} '
#             f'{args.obj} '
#             '--force_cuda_rast')
