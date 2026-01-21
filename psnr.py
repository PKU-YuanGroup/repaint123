# eval the clip-similarity for an input image and a geneated mesh
import cv2
import torch
import numpy as np
from torchvision import transforms as T
# from transformers import CLIPModel, CLIPTokenizer, CLIPProcessor

import kiui
from kiui.render import GUI
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from cam_utils import orbit_camera
from mesh_renderer import Renderer

# import debugpy; debugpy.connect(("localhost", 5677)) 


# class CLIP:
#     def __init__(self, device, model_name='openai/clip-vit-large-patch14'):

#         self.device = device

#         self.clip_model = CLIPModel.from_pretrained(model_name).to(self.device)
#         self.processor = CLIPProcessor.from_pretrained(model_name)
    
#     def encode_image(self, image):
#         # image: PIL, np.ndarray uint8 [H, W, 3]

#         pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.to(self.device)
#         image_features = self.clip_model.get_image_features(pixel_values=pixel_values)

#         image_features = image_features / image_features.norm(dim=-1,keepdim=True)  # normalize features

#         return image_features

#     def encode_text(self, text):
#         # text: str

#         inputs = self.processor(text=[text], padding=True, return_tensors="pt")
#         text_features = self.clip_model.get_text_features(**inputs)

#         text_features = text_features / text_features.norm(dim=-1,keepdim=True)  # normalize features

#         return text_features


if __name__ == '__main__':
    import os
    import tqdm
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('image', type=str, help="path to front view image")
    parser.add_argument('mesh', type=str, help="path to mesh (obj, glb, ...)")
    parser.add_argument('--front_dir', type=str, default='+z', help="mesh front-facing dir")
    parser.add_argument('--mode', default='albedo', type=str, choices=['lambertian', 'albedo', 'normal', 'depth'], help="rendering mode")
    parser.add_argument('--W', type=int, default=800, help="GUI width")
    parser.add_argument('--H', type=int, default=800, help="GUI height")
    parser.add_argument('--radius', type=float, default=2, help="default GUI camera radius from center")
    parser.add_argument('--fovy', type=float, default=49.1, help="default GUI camera fovy")
    parser.add_argument('--elevation', type=int, default=0, help="rendering elevation")
    parser.add_argument('--num_azimuth', type=int, default=8, help="number of images to render from different azimuths")
    parser.add_argument('--force_cuda_rast', action="store_true", help="force cuda rasterizer")
    parser.add_argument('--train_geo', action="store_true", help="")
    
    opt = parser.parse_args()
    opt.wogui = True

    # clip = CLIP('cuda')
    # clip = CLIP('cuda', model_name='laion/CLIP-ViT-bigG-14-laion2B-39B-b160k')
    
    gui = GUI(opt)
    gui.renderer = Renderer(opt)

    # load image and encode as ref features
    ref_img = kiui.read_image(opt.image, mode='float')
    if ref_img.shape[-1] == 4:
        # rgba to white-bg rgb
        ref_img = ref_img[..., :3] * ref_img[..., 3:] + (1 - ref_img[..., 3:])
    ref_img = (ref_img * 255).astype(np.uint8)
    ref_img = cv2.resize(ref_img, (opt.H, opt.W), interpolation=cv2.INTER_AREA)
    # with torch.no_grad():
    #     ref_features = clip.encode_image(ref_img)

    elevation = 0
    azimuth = 0
    gui.cam.from_angle(elevation, azimuth)
    gui.need_update = True
    gui.step()
    image = (gui.render_buffer * 255).astype(np.uint8)
    
    # ssaa = min(2.0, max(0.125, 2 * np.random.random()))
    # out = gui.renderer.render(orbit_camera(0, 0, opt.radius), gui.cam.perspective, opt.H, opt.W, ssaa=ssaa)
    out = gui.renderer.render(orbit_camera(0, 0, opt.radius), gui.cam.perspective, opt.H, opt.W)
    
    
    # calculate psnr of image and ref_img
    # psnr = image.astype(np.float32) - ref_img.astype(np.float32)
    # psnr = np.mean(psnr ** 2)
    # psnr = 10 * np.log10(255 ** 2 / psnr)
    # print(psnr)
    
    psnr = compare_psnr(
                out["image"].mul(255).to(torch.uint8).cpu().numpy(), ref_img,
                data_range=255)
    print("psnr: ", psnr)
    
    
    
    # for ele in tqdm.tqdm(elevation):
    #     for azi in tqdm.tqdm(azimuth):
    #         gui.cam.from_angle(ele, azi)
    #         gui.need_update = True
    #         gui.step()
    #         image = (gui.render_buffer * 255).astype(np.uint8)
    #         with torch.no_grad():
    #             cur_features = clip.encode_image(image)
            
    #         # kiui.lo(ref_features, cur_features)
    #         similarity = (ref_features * cur_features).sum(dim=-1).mean().item()

    #         results.append(similarity)
    
    # avg_similarity = np.mean(results)
    
    # print(avg_similarity)

            