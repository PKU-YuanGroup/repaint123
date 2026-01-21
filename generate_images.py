# eval the clip-similarity for an input image and a geneated mesh
import cv2
import torch
import numpy as np
from torchvision import transforms as T
from transformers import CLIPModel, CLIPTokenizer, CLIPProcessor

import kiui
from kiui.render import GUI
from cam_utils import orbit_camera
from mesh_renderer import Renderer
from PIL import Image

if __name__ == '__main__':
    import os
    import tqdm
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('image', type=str, help="path to front view image")
    parser.add_argument('mesh', type=str, help="path to mesh (obj, glb, ...)")
    parser.add_argument('--out', type=str, help="path to mesh (obj, glb, ...)")
    parser.add_argument('--front_dir', type=str, default='+z', help="mesh front-facing dir")
    parser.add_argument('--mode', default='albedo', type=str, choices=['lambertian', 'albedo', 'normal', 'depth'], help="rendering mode")
    parser.add_argument('--W', type=int, default=800, help="GUI width")
    parser.add_argument('--H', type=int, default=800, help="GUI height")
    parser.add_argument('--radius', type=float, default=2, help="default GUI camera radius from center")
    parser.add_argument('--fovy', type=float, default=49.1, help="default GUI camera fovy")
    parser.add_argument('--elevation', type=int, default=0, help="rendering elevation")
    parser.add_argument('--num_azimuth', type=int, default=100, help="number of images to render from different azimuths")
    parser.add_argument('--force_cuda_rast', action="store_true", help="force cuda rasterizer")
    parser.add_argument('--train_geo', action="store_true", help="")
    opt = parser.parse_args()
    opt.wogui = True
    
    gui = GUI(opt)
    gui.renderer = Renderer(opt)
    
    out_dir = opt.out
    os.makedirs(out_dir, exist_ok=True)

    # load image and encode as ref features
    # ref_img = kiui.read_image(opt.image, mode='float')
    # if ref_img.shape[-1] == 4:
    #     # rgba to white-bg rgb
    #     ref_img = ref_img[..., :3] * ref_img[..., 3:] + (1 - ref_img[..., 3:])
    # ref_img = (ref_img * 255).astype(np.uint8)
    # ref_img = cv2.resize(ref_img, (opt.H, opt.W), interpolation=cv2.INTER_AREA)

    elevation = [opt.elevation,]
    azimuth = np.linspace(0, 360, opt.num_azimuth, dtype=np.int32, endpoint=True)
    for ele in tqdm.tqdm(elevation):
        for azi in tqdm.tqdm(azimuth):
            gui.cam.from_angle(ele, azi)
            gui.need_update = True
            gui.step()
            
            # ssaa = min(2.0, max(0.125, 2 * np.random.random()))
            # out = gui.renderer.render(orbit_camera(ele, azi, opt.radius), gui.cam.perspective, opt.H, opt.W, ssaa=ssaa)
            out = gui.renderer.render(orbit_camera(ele, azi, opt.radius), gui.cam.perspective, opt.H, opt.W)
            image = out["image"].mul(255).to(torch.uint8).cpu().numpy()
            Image.fromarray(image).save(f'{out_dir}/{ele}_{azi}.png')
            # Image.fromarray(out["viewcos"].mul(255).to(torch.uint8).cpu().numpy()).save(f'{out_dir}/{ele}_{azi}_normal.png')


            