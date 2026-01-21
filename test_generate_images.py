import os
import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dir', default='data2', type=str, help='Directory where processed images are stored')
parser.add_argument('--out', default='logs_realfusion_ours_rmedge_soft3_1loss_5002', type=str, help='Directory where obj files will be saved')
parser.add_argument('--dataset', default='realfusion15', type=str, help='Directory where obj files will be saved')
parser.add_argument('--gpu', default=0, type=int, help='ID of GPU to use')
parser.add_argument('--elevation', default=0, type=int, help='Elevation angle of view in degrees')
parser.add_argument('--config', default='configs', type=str, help='Path to config directory, which contains image_realfusion.yaml')
args = parser.parse_args()

files = glob.glob(f'{args.dir}/*_rgba.png')
print(len(files))
configs_dir = args.config

# create output directories if not exists
out_dir = args.out
os.makedirs(out_dir, exist_ok=True)
out_dir_image = f'results/{out_dir}/{args.dataset}'
os.makedirs(out_dir_image, exist_ok=True)

for file in files:
    name = os.path.basename(file).replace("_rgba.png", "")
    os.system(f'CUDA_VISIBLE_DEVICES={args.gpu} python generate_images.py '
              f'{file} '
              f'{out_dir}/{name}/{name}.obj '
              f'--out {out_dir_image}/{name} '
              '--force_cuda_rast')