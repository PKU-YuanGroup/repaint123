import os
import glob
import argparse
import kiui
parser = argparse.ArgumentParser()
parser.add_argument('--dir', default='./', type=str, help='Directory where obj files are stored')
parser.add_argument('--out', default='./videos', type=str, help='Directory where videos will be saved')
args = parser.parse_args()

out = args.out
os.makedirs(out, exist_ok=True)

files = glob.glob(f'{args.dir}/*.obj')
for f in files:
    name = os.path.basename(f)
    # first stage model, ignore
    if name.endswith('_mesh.obj'): 
        continue
    print(f'[INFO] process {name}')
    #os.system(f"CUDA_VISIBLE_DEVICES=1 python -m kiui.render {f} --save './images' --num_azimuth 9 ")
    os.system(f"python -m kiui.render {f} --save_video {os.path.join(out, name.replace('.obj', '.mp4'))} ")

    