import os
import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dir",
    default="datasets/alpha",
    type=str,
    help="Directory where processed images are stored",
)
parser.add_argument(
    "--out",
    default="logs_alpha_ours_rmedge_soft3_1loss_ip2_masa_resize_inv0.6_ip0.5",
    type=str,
    help="Directory where obj files will be saved",
)
parser.add_argument(
    "--video-out",
    default="videos_alpha_ours_rmedge_soft3_1loss_ip2_masa_resize",
    type=str,
    help="Directory where videos will be saved",
)
parser.add_argument("--gpu", default=0, type=int, help="ID of GPU to use")
parser.add_argument(
    "--elevation", default=0, type=int, help="Elevation angle of view in degrees"
)
parser.add_argument(
    "--config",
    default="configs",
    type=str,
    help="Path to config directory, which contains image_realfusion.yaml",
)
parser.add_argument(
    "--fine_config_name",
    default="image2",
    type=str,
    help="Path to config directory, which contains image_realfusion.yaml",
)
parser.add_argument("--ddnm_strength", default=0.4, type=float, help="")
args = parser.parse_args()

files = glob.glob(f"{args.dir}/*_rgba.png")
configs_dir = args.config

# check if image_realfusion.yaml exists
if not os.path.exists(os.path.join(configs_dir, "image2.yaml")):
    raise FileNotFoundError(
        f"image2.yaml not found in {configs_dir} directory. Please check if the directory is correct."
    )

# create output directories if not exists
out_dir = args.out
os.makedirs(out_dir, exist_ok=True)
video_dir = args.video_out
os.makedirs(video_dir, exist_ok=True)

print(args.fine_config_name)

for file in files:
    name = os.path.basename(file).replace("_rgba.png", "")
    if os.path.exists(os.path.join(video_dir, f"{name}.mp4")):
        continue
    print(f"======== processing {name} ========")
    # # first stage
    os.system(
        f"CUDA_VISIBLE_DEVICES={args.gpu} python main.py "
        f"--config {configs_dir}/image_realfusion.yaml "
        f"input={file} "
        f"outdir={out_dir}/{name} "
        f"save_path={name} elevation={args.elevation}"
    )
    # second stage
    os.system(
        f"CUDA_VISIBLE_DEVICES={args.gpu} python main2.py "
        f"--ddnm_strength {args.ddnm_strength} "
        f"--config {configs_dir}/{args.fine_config_name}.yaml "
        f"input={file} "
        f"outdir={out_dir}/{name} "
        f"save_path={name} elevation={args.elevation}"
    )
    # export video
    mesh_path = os.path.join(out_dir, name, f"{name}.obj")
    os.system(
        f"CUDA_VISIBLE_DEVICES={args.gpu} python -m kiui.render {mesh_path} "
        f"--save_video {video_dir}/{name}.mp4 "
        f"--wogui "
        f"--force_cuda_rast "
        f"--elevation {args.elevation}"
    )
