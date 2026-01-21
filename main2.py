import os
import cv2
import time
import tqdm
import numpy as np
import dearpygui.dearpygui as dpg

import torch
import torch.nn.functional as F

import trimesh
import rembg

from cam_utils import orbit_camera, OrbitCamera
from mesh_renderer import Renderer

from cam_utils import get_ray_directions
from PIL import Image
import json
from diffusers import DDIMScheduler
from inpaint.views_dataset import MultiviewDataset
from torch.utils.data import DataLoader
from typing import Any, Dict, Union, List
from pathlib import Path
from inpaint.view_generation.depth_supervised_inpainting_pipeline import (
    StableDiffusionDepth2ImgInpaintingPipeline,
)
from inpaint.view_generation.MasaCtrlPipeline import(
    MasaCtrlPipeline,
)
# from inpaint.view_generation.ControlNetPipeline import(
#     ControlNetPipeline,
# )
from inpaint.view_generation.ControlNetPipeline import ControlNetMasaCachePipeline2 as ControlNetPipeline
from diffusers import (
    ControlNetModel,
)
from inpaint.view_generation.masactrl_utils2 import(
    AttentionBase, regiter_attention_editor_diffusers
)
# from inpaint.view_generation.masactrl import(
#     MutualSelfAttentionControl,
# )
from inpaint.view_generation.masactrl import MutualSelfAttentionControlCache as MutualSelfAttentionControl
from inpaint.view_generation.mask_options import (
    mask_proc_options,
    mask_options,
)
from inpaint.view_generation.utils import import_config_key, view_dep_prompt

from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
)
from ip_adapter.ip_adapter import IPAdapter
import torchvision.transforms.functional as TF
from torchvision.utils import make_grid

import debugpy; debugpy.connect(("localhost", 5678)) 
from transformers import AutoTokenizer, AutoModel
import torch
import torchvision.transforms as T
from PIL import Image

from torchvision.transforms.functional import InterpolationMode


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_file, input_size=448, max_num=6):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values



# from kiui.lpips import LPIPS
      
def view_dep_prompt(prompt, angle, color=""):
    if prompt == "":
        base_prompt = "A photo"
    else:
        if color == "":
            base_prompt = f"A photo of a {prompt}"
        else:
            base_prompt = f"A photo of a {color} {prompt}"
    if angle <= 45 or angle >= 315:
        return f"{base_prompt}, front view"
    if 45 <= angle <= 135:
        return f"{base_prompt}, left view"
    if 135 <= angle <= 225:
        return f"{base_prompt}, back view"
    if 225 <= angle <= 315:
        return f"{base_prompt}, right view"
    return prompt

def nonzero_normalize_depth(depth, mask=None):
    if mask is not None:
        if (depth[mask]>0).sum() > 0:
            nonzero_depth_min = depth[mask][depth[mask]>0].min()
        else:
            nonzero_depth_min = 0
    else:
        if (depth>0).sum() > 0:
            nonzero_depth_min = depth[depth>0].min()
        else:
            nonzero_depth_min = 0
    if nonzero_depth_min == 0:
        return depth
    else:
        depth = (depth - nonzero_depth_min) / depth.max()
        return depth.clamp(0, 1)

def save_tensor2image(x: torch.Tensor, path, channel_last=False, quality=75, **kwargs):
    # assume the input x is channel last
    if x.ndim == 4 and channel_last:
        x = x.permute(0, 3, 1, 2) 
    TF.to_pil_image(make_grid(x, value_range=(0, 1), **kwargs)).save(path, quality=quality)
    
class GUI:
    def __init__(self, opt):
        self.opt = opt  # shared with the trainer's opt to support in-place modification of rendering parameters.
        self.gui = opt.gui # enable gui
        self.W = opt.W
        self.H = opt.H
        self.cam = OrbitCamera(opt.W, opt.H, r=opt.radius, fovy=opt.fovy)

        self.mode = "image"
        self.seed = "random"

        self.buffer_image = np.ones((self.W, self.H, 3), dtype=np.float32)
        self.need_update = True  # update buffer_image

        # models
        self.device = torch.device("cuda")
        self.bg_remover = None

        self.guidance_sd = None
        self.guidance_zero123 = None

        self.enable_sd = False
        self.enable_zero123 = False

        # renderer
        self.renderer = Renderer(opt).to(self.device)

        # input image
        self.input_img = None
        self.input_mask = None
        self.input_img_torch = None
        self.input_mask_torch = None
        self.overlay_input_img = False
        self.overlay_input_img_ratio = 0.5

        # input text
        self.prompt = ""
        self.negative_prompt = ""

        # training stuff
        self.training = False
        self.optimizer = None
        self.step = 0
        self.train_steps = 1  # steps per rendering loop
        # self.lpips_loss = LPIPS(net='vgg').to(self.device)
        
        # load input data from cmdline
        if self.opt.input is not None:
            self.load_input(self.opt.input)
        
        # override prompt from cmdline
        if self.opt.prompt is not None:
            self.prompt = self.opt.prompt
        
        if self.gui:
            dpg.create_context()
            self.register_dpg()
            self.test_step()

    def __del__(self):
        if self.gui:
            dpg.destroy_context()

    def seed_everything(self):
        try:
            seed = int(self.seed)
        except:
            seed = np.random.randint(0, 1000000)

        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

        self.last_seed = seed

    def prepare_train(self):

        self.step = 0

        # setup training
        self.optimizer = torch.optim.Adam(self.renderer.get_params())

        # default camera
        pose = orbit_camera(self.opt.elevation, 0, self.opt.radius)
        self.fixed_cam = [(pose, self.cam.perspective)]
        

        self.enable_sd = self.opt.lambda_sd > 0
        # self.enable_sd = self.opt.lambda_sd > 0 and self.prompt != ""
        self.enable_zero123 = self.opt.lambda_zero123 > 0 and self.input_img is not None

        # lazy load guidance model
        if self.guidance_sd is None and self.enable_sd:
            if self.opt.mvdream:
                print(f"[INFO] loading MVDream...")
                from guidance.mvdream_utils import MVDream
                self.guidance_sd = MVDream(self.device)
                print(f"[INFO] loaded MVDream!")
            else:
                print(f"[INFO] loading SD...")
                from guidance.sd_utils import StableDiffusion
                self.guidance_sd = StableDiffusion(self.device)
                print(f"[INFO] loaded SD!")

        if self.guidance_zero123 is None and self.enable_zero123:
            print(f"[INFO] loading zero123...")
            from guidance.zero123_utils import Zero123
            self.guidance_zero123 = Zero123(self.device)
            print(f"[INFO] loaded zero123!")

        # input image
        if self.input_img is not None:
            self.update_ref()

        # prepare embeddings
        self.prepare_embed()
        
        if self.opt.iterative:
            self.prepare_inpaint()
        
    def prepare_embed(self):
        with torch.no_grad():

            if self.enable_sd:
                self.guidance_sd.get_text_embeds([self.prompt], [self.negative_prompt])

            if self.enable_zero123:
                self.guidance_zero123.get_img_embeds(self.input_img_torch)
        
    def prepare_inpaint(self):
        # config
        with open(self.opt.inpaint_config_path) as json_file:
            self.inpaint_config = json.load(json_file)
        self.paint_step = 0
        angles = np.arange(0, 360, self.inpaint_config["angle_inc"])
        self.num_views = len(angles)
        self.interval = self.opt.iters_refine / (self.num_views+1)
        if self.num_views % 2 == 0:
            self.angles = [angles[0]] + [i for j in zip(angles[1:self.num_views // 2], angles[-1:self.num_views // 2:-1]) for i in j] + [angles[self.num_views // 2]]
        else:
            self.angles = [angles[0]] + [i for j in zip(angles[1:self.num_views // 2 + 1], angles[-1:self.num_views // 2:-1]) for i in j]
        self.debug_path = os.path.join(self.opt.outdir, 'debug')
        os.makedirs(self.debug_path, exist_ok=True)
        # os.makedirs(self.opt.ref_path, exist_ok=True)
        self.flag = True
        self.recon = False
        self.occlu_mask = []
        
        # pipeline
        controlnet_depth = ControlNetModel.from_pretrained("/data/aigc/tzy/zjw/control_v11f1p_sd15_depth").to("cuda")
        self.scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012,
                          beta_schedule="scaled_linear", clip_sample=False,
                          set_alpha_to_one=False, steps_offset=1)
        self.pipe = ControlNetPipeline.from_pretrained(
               "/data/aigc/tzy/zjw/runwayml-stable-diffusion-v1-5", controlnet=controlnet_depth, scheduler= self.scheduler,
        ).to("cuda")
        
        # inpaint config
        rng_randn = torch.Generator(device=self.device)
        self.latents = torch.randn(
            (1, 4, self.opt.ref_size // 8, self.opt.ref_size // 8), generator=rng_randn, dtype=torch.float16, device=self.device
        )
        self.num_inference_steps = self.inpaint_config["num_inference_steps"]
        self.n_propmt = import_config_key(self.inpaint_config, "negative_prompt", None)
        self.inpainting_strength = import_config_key(self.inpaint_config, "inpainting_strength", 1)
        self.mask_option = self.inpaint_config["mask_blend"]
        self.mask_blend_kernel = import_config_key(self.inpaint_config, "mask_blend_kernel", -1)
        self.latent_blend_kernel = import_config_key(self.inpaint_config, "latent_blend_kernel", -1)
        
        image_encoder_path = "models/image_encoder"
        ip_ckpt = "models/ip-adapter-plus_sd15.bin"
        self.ip_adapter = IPAdapter(self.pipe, ip_ckpt, image_encoder_path, scale=self.opt.scale, device=self.device)
        # self.ip_adapter = IPAdapter(self.pipe, ip_ckpt, image_encoder_path, device=self.device)

        step = 0
        layer = 10
        self.masa_editor = MutualSelfAttentionControl(step, layer, total_steps=self.num_inference_steps, strength=self.opt.denoise_strength)
        self.base_editor = AttentionBase()
        
        path = "~/tzy/zjw/OpenGVLab/InternVL-Chat-V1-5"
        # If you have an 80G A100 GPU, you can put the entire model on a single GPU.
        self.caption_model = AutoModel.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True).eval().cuda()
        # Otherwise, you need to set device_map='auto' to use multiple GPUs for inference.
        # import os
        # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        # model = AutoModel.from_pretrained(
        #     path,
        #     torch_dtype=torch.bfloat16,
        #     low_cpu_mem_usage=True,
        #     trust_remote_code=True,
        #     device_map='auto').eval()

        self.caption_tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        
        # set the max number of tiles in `max_num`
        pixel_values = self.input_img_torch[0].to(torch.bfloat16).cuda()

        generation_config = dict(
            num_beams=1,
            max_new_tokens=512,
            do_sample=False,
        )


        # single-round single-image conversation
        question = "Describe the image in detail." # Please describe the picture in detail
        response = self.caption_model.chat(self.caption_tokenizer, pixel_values, question, generation_config)
        self.text_ref = response
        
    # paint config
    # paint facade
    # paint bidirectional
    # connect to init
    # connect to train_step
    
    @torch.no_grad()
    def normed_depth_2_absolute_depth(self, depth_tensors):
        Z_near, Z_far = self.cam.near, self.cam.far
        scale, bias = -(Z_far+Z_near)/(Z_far-Z_near), -(2*Z_far*Z_near)/(Z_far-Z_near)
        depth_tensors_new = []
        for depth_tensor in depth_tensors:
            depth_tensor = torch.where(depth_tensor == 0, torch.ones_like(depth_tensor), depth_tensor)
            depth_tensor = bias/(depth_tensor+scale)
            depth_tensors_new.append(depth_tensor)
        return depth_tensors_new
    
    @torch.no_grad()
    def get_occlu_mask(self, depth_tensor, depth_tensor2, camera):
        Z_near, Z_far = self.cam.near, self.cam.far
        # depth_tensor_abs, depth_tensor2_abs = self.normed_depth_2_absolute_depth([depth_tensor, depth_tensor2])
        depth_tensor_abs, depth_tensor2_abs = depth_tensor, depth_tensor2
        front_mask = depth_tensor != 0
        
        def convert_depth_to_ptcloud(depth):
            depth = depth.reshape(
                depth.shape[0], depth.shape[1], 1
            )
            depth = torch.where(
                depth == 0,
                Z_far * torch.ones_like(depth),
                depth,
            )
            H, W = depth.shape[0:2]
            focal_length = [
                1 / np.tan(self.opt.fovy / 2 * np.pi / 180) * H / 2,
                1 / np.tan(self.opt.fovy / 2 * np.pi / 180) * W / 2,
            ]
            ray_directions = get_ray_directions(H, W, focal_length).to(self.device)
            pt_cloud = ray_directions * depth
            pt_cloud[:, :, 2] = pt_cloud[:, :, 2] - self.cam.radius
            return pt_cloud
        
        pt_cloud2 = convert_depth_to_ptcloud(depth_tensor2_abs)
        pt_cloud = convert_depth_to_ptcloud(depth_tensor_abs)
        surface_pts = camera.get_world_to_view_transform().transform_points(pt_cloud)
        surface_pts[:, :, 2] = surface_pts[:, :, 2] - self.cam.radius
        surface_pts_masked = surface_pts * front_mask

        ndc_coords = -camera.transform_points_ndc(pt_cloud)[:, :, :2]
        grid = ndc_coords[None, ...]
        rev_depth = torch.nn.functional.grid_sample(pt_cloud2[None,].permute(0, 3, 1, 2), grid, mode="bilinear", padding_mode="zeros")
        rev_depth_masked = rev_depth.permute(0,2,3,1)[0] * front_mask
        
        occlu_threshold = self.opt.occlu_threshold
        occlu_mask = torch.sum((surface_pts_masked[:, :, 2:3]- rev_depth_masked[:, :, 2:3])**2, axis=-1) > occlu_threshold**2
        
        return occlu_mask, grid, surface_pts_masked, ndc_coords
    
    @torch.no_grad()
    def backward_oculusion_aware_render(self, out, out2, elevation_inc, azimuth_inc, rm_normal=False, rm_edge=False):
        # get camera
        elev_angles = torch.tensor([elevation_inc]).to(self.device)
        azim_angles = torch.tensor([-azimuth_inc], dtype=torch.float32).to(self.device)
        R, T = look_at_view_transform(
            self.cam.radius, elev_angles.flatten(), (azim_angles + 180).flatten()
        )
        cam = FoVPerspectiveCameras(znear=self.cam.near, zfar=self.cam.far, device=self.device, R=R, T=T, fov=np.rad2deg(self.cam.fovy))
        
        # get occlusion mask
        depth, depth2 = out['depth'], out2['depth']
        if rm_edge:
            mask = out2['alpha'] > 0
            mask = torch.from_numpy(self.rm_edge(mask.cpu().numpy())).to(depth2.device)
            depth2 = depth2 * mask.float()
            
        if rm_normal:
            valid_mask = out2["viewcos"] > 0.5
            depth2 = depth2 * valid_mask.float()
        occlu_mask, grid, surface_pts_masked, ndc_coords = self.get_occlu_mask(depth, depth2, cam)
        # Image.fromarray(occlu_mask.to(torch.uint8).mul(255).cpu().numpy()).save(os.path.join(self.debug_path, f'{self.paint_step}_v{view}_v{view2}_mask.jpg'))
        
        z_normal = (out['viewcos']* (out['alpha'] > 0).float()).permute(2,0,1)[None]
        rev_z_normal = torch.nn.functional.grid_sample(out2['viewcos'].permute(2,0,1)[None], grid, mode="nearest", padding_mode="zeros")
        rev_z_normal = rev_z_normal * (out['alpha'] > 0).float().permute(2,0,1)[None]
        
        z = rev_z_normal - z_normal
        z[z>=0] = 1 
        z[z<0] = rev_z_normal[z<0] / z_normal[z<0]
        
        return occlu_mask, grid, surface_pts_masked, ndc_coords, z
    
    @torch.no_grad()
    def ddim_inv(self, image_tensor, azimuth):
        print("ddim inversion")

        self.base_editor.reset()
        regiter_attention_editor_diffusers(self.pipe, self.base_editor)
        
        image = self.tensor2img(image_tensor)
        start_code, latents_list, noise_list = self.pipe.invert(
            # prompt= "",
            # negative_prompt = "",
            prompt_embeds = self.negative_prompt_embeds,
            negative_prompt_embeds = self.negative_prompt_embeds,
            image= image,
            strength=self.opt.denoise_strength,
            num_inference_steps=self.num_inference_steps,
            latents=self.latents,
            return_intermediates= True,
            desc=f"Invert {azimuth} deg. view",
            ddnm=True
        )
        
        return start_code, latents_list, noise_list
    
    def update_ref(self):
        # downsample
        img = self.input_img
        img = (img[..., :3] * self.input_mask * 255).astype(np.uint8)
        img = np.concatenate([img, self.input_mask * 255], axis=-1)
        img = cv2.resize(
            img, (self.opt.ref_size, self.opt.ref_size), interpolation=cv2.INTER_AREA
        )
        img = img.astype(np.float32) / 255.0
        self.input_img = img[..., :3] * img[...,3:] + (
            1 - img[...,3:]
        )
        
        self.input_img_torch_ = torch.from_numpy(self.input_img).permute(2, 0, 1).unsqueeze(0).to(self.device)
        # self.input_img_torch_ = F.interpolate(self.input_img_torch_, (self.opt.ref_size, self.opt.ref_size), mode="bilinear", align_corners=False)
        if self.input_img_torch == None:
            self.input_img_torch = self.input_img_torch_
        else:
            self.input_img_torch = torch.cat([self.input_img_torch, self.input_img_torch_])
            
        self.input_mask_torch_ = torch.from_numpy(self.input_mask).permute(2, 0, 1).unsqueeze(0).to(self.device)
        self.input_mask_torch_ = F.interpolate(self.input_mask_torch_, (self.opt.ref_size, self.opt.ref_size), mode="bilinear", align_corners=False)
        if self.input_mask_torch == None:
            self.input_mask_torch = self.input_mask_torch_
        else:
            self.input_mask_torch = torch.cat([self.input_mask_torch, self.input_mask_torch_])
    
    def save_update_ref(self, image_np, mask_np, img_path, angle):
        pose = orbit_camera(self.opt.elevation, angle, self.opt.radius)
        self.fixed_cam += [(pose, self.cam.perspective)]
        rgba = np.concatenate([image_np, mask_np.astype(np.uint8)*255], axis=-1)
        Image.fromarray(rgba).save(img_path)
        self.load_input(img_path)
        self.update_ref()
        
    @torch.no_grad()
    def inpaint(self, output, mask_image, azimuth, start_code, latents_list):
        print("inpaint")
        depth = nonzero_normalize_depth(output['depth'].permute(2,0,1)[None])
        depth[depth>0] = 1 - depth[depth>0]
        depth = Image.fromarray(depth[0,0].mul(255).to(torch.uint8).cpu().numpy()).convert('RGB')
        depth.save(os.path.join(self.debug_path, f'{self.paint_step}_{azimuth}_depth.jpg'))
        
        input_image = Image.fromarray(output['image'].mul(255).to(torch.uint8).cpu().numpy())
        

        print(view_dep_prompt(self.text_ref, azimuth), f'azimuth: {azimuth}')
        text_embeds = self.pipe.encode_prompt(
                view_dep_prompt(self.text_ref, azimuth),
                negative_prompt='',
                device=self.pipe.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True
            )
        text_length = text_embeds[0].shape[1]
        prompt_embeds = torch.cat((text_embeds[0], self.prompt_embeds[:,text_length:]), dim=1)
        negative_prompt_embeds = torch.cat((text_embeds[1], self.negative_prompt_embeds[:,text_length:]), dim=1)
        
        # tar_image = Image.open("/data/aigc/tzy/zjw/dreamgaussian/logs_repaint/bird_2_ip_500/debug/3_80_input_img.jpg").convert("RGB")
        # start_code_1, latents_list_1, noise_list_1 = self.ddim_inv(tar_image, self.angles[1])
        
        self.masa_editor.reset()
        regiter_attention_editor_diffusers(self.pipe, self.masa_editor)
        image = self.pipe(
            # prompt=['blue bird', 'blue bird'],
            # negative_prompt=[""] * 2,
            # prompt_embeds = torch.cat([self.prompt_embeds]*2 ),
            # negative_prompt_embeds = torch.cat([self.negative_prompt_embeds]*2),
            prompt_embeds = prompt_embeds,
            negative_prompt_embeds = negative_prompt_embeds,
            # image=[self.input_img_torch[:1], output['image'].permute(2,0,1)[None]],
            image=input_image,
            latent_code =start_code,
            latent_list = latents_list, 
            mask_image=mask_image,
            control_image= depth,
            # controlnet_conditioning_scale = 0,
            strength=self.opt.denoise_strength,
            num_inference_steps=self.num_inference_steps,
            latents=self.latents,
            inpainting_strength=self.inpainting_strength,
            mask_blend_kernel=self.mask_blend_kernel,
            # latent_blend_kernel=self.latent_blend_kernel,
            align=False,
            align_var=False,
            noise_list = self.front_noise_list,
            desc=f"Inpainting {azimuth} deg. view",
            ddnm=True,
            ddnm_strength=self.opt.ddnm_strength,
            repeat=1
        ).images[0]
        return image
    
    @torch.no_grad()
    def set_kvcache(self):
        self.masa_editor.reset()
        regiter_attention_editor_diffusers(self.pipe, self.masa_editor, hook=True)
        image = self.pipe(
            # prompt=['blue bird', 'blue bird'],
            # negative_prompt=[""] * 2,
            # prompt_embeds = torch.cat([self.prompt_embeds]*2 ),
            # negative_prompt_embeds = torch.cat([self.negative_prompt_embeds]*2),
            prompt_embeds = self.prompt_embeds,
            negative_prompt_embeds = self.negative_prompt_embeds,
            # image=[self.input_img_torch[:1], output['image'].permute(2,0,1)[None]],
            image=self.ref_img,
            latent_code =self.front_start_code,
            replace_latent_list = self.front_latents_list, 
            # mask_image=mask_image,
            control_image= self.ref_depth,
            # controlnet_conditioning_scale = 0,
            strength=self.opt.denoise_strength,
            num_inference_steps=self.num_inference_steps,
            latents=self.latents,
            inpainting_strength=self.inpainting_strength,
            mask_blend_kernel=self.mask_blend_kernel,
            # latent_blend_kernel=self.latent_blend_kernel,
            align=False,
            align_var=False,
            noise_list = self.front_noise_list,
            desc=f"KV cache 0 deg. view",
        ).images[0]
        return image
    
    def render_view(self, elevation, azimuth):
        pose = orbit_camera(elevation, azimuth, self.cam.radius)
        cam = (pose, self.cam.perspective)
        outputs = self.renderer.render(*cam, self.opt.ref_size, self.opt.ref_size)
        return outputs
    
    def prepare_mask(self,occlu_mask_tensor):
        occlu_mask_np = occlu_mask_tensor.to(torch.uint8).mul(255).cpu().numpy()
        keep_mask_np = mask_proc_options[3](255-occlu_mask_np, kernel_size=5)
        keep_mask = torch.from_numpy(keep_mask_np/255).float()
        keep_mask = F.interpolate(keep_mask[None,None], (self.opt.ref_size//8, self.opt.ref_size//8), mode="nearest")
        return keep_mask
    
    def tensor2img(self, img_tensor):
        img = Image.fromarray(img_tensor[0].permute(1,2,0).mul(255).to(torch.uint8).cpu().numpy())
        return img
    
    def update_prev_occlu_mask(self, index, soft_mask, grid, obj_mask):
        H, W = self.opt.ref_size, self.opt.ref_size
        soft_mask_diff = (soft_mask<1) & (soft_mask>0)
        soft_mask_diff = torch.nn.functional.interpolate(soft_mask_diff.float(), (H, W), mode="nearest")
        soft_mask_diff = soft_mask_diff * obj_mask
        soft_mask_diff = (soft_mask_diff > 0.5).to(self.device)
        grid[...,0] = (grid[...,0]+1)*H/2
        grid[...,1] = (grid[...,1]+1)*W/2
        indices = grid.long().detach()
        indices = indices[soft_mask_diff[0,...,None].expand_as(indices)].reshape(-1,2)
        if indices.shape[0] != 0:
            white_bg = 255*np.ones((H, W), dtype=np.uint8)
            indices_ = indices.cpu()
            white_bg[indices_[...,1], indices_[...,0]] = 0
            Image.fromarray(white_bg).save(os.path.join(self.debug_path, f'{self.paint_step}_prev_occlu_mask{index}.jpg'))
            # perform open morphologyEx
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            mask_open = cv2.morphologyEx(white_bg, cv2.MORPH_OPEN, kernel)
            Image.fromarray(mask_open).save(os.path.join(self.debug_path, f'{self.paint_step}_prev_occlu_mask{index}_open.jpg'))
            mask_open = torch.from_numpy(mask_open > 128)[None,None].to(self.device)
            self.occlu_mask[index] &= mask_open
            
        # if indices.shape[0] != 0:
        #     self.occlu_mask[index][0,0][indices[...,1], indices[...,0]] = False
                
    @torch.no_grad()
    def paint_facade(self):
        # ========== 1. render views ==========
        outputs = self.render_view(self.opt.elevation, self.angles[0])
        outputs1 = self.render_view(self.opt.elevation, self.angles[1])
        outputs2 = self.render_view(self.opt.elevation, self.angles[2])
        depth = nonzero_normalize_depth(outputs['depth'].permute(2,0,1)[None])
        depth[depth>0] = 1 - depth[depth>0]
        self.ref_depth = Image.fromarray(depth[0,0].mul(255).to(torch.uint8).cpu().numpy()).convert('RGB')
        self.ref_depth.save(os.path.join(self.debug_path, f'{self.paint_step}_ref_depth.jpg'))
        self.ref_img = self.tensor2img(self.input_img_torch[:1])
        
        # ========== 2. caculate inpainting masks ==========
        occlu_mask1, grid1, surface_pts_masked1, ndc_coords1, soft_mask1 = self.backward_oculusion_aware_render(
            outputs1, outputs, 0, self.angles[1]-self.angles[0], rm_edge=True)
        occlu_mask2, grid2, surface_pts_masked2, ndc_coords2, soft_mask2 = self.backward_oculusion_aware_render(
            outputs2, outputs, 0, self.angles[2]-self.angles[0], rm_edge=True)
        occlu_mask_img1 = Image.fromarray(occlu_mask1.to(torch.uint8).mul(255).cpu().numpy())
        occlu_mask_img2 = Image.fromarray(occlu_mask2.to(torch.uint8).mul(255).cpu().numpy())
        obj_mask1 = self.prepare_mask(outputs1['alpha'][...,0]>0)
        obj_mask2 = self.prepare_mask(outputs2["alpha"][...,0]>0)
        keep_mask1 = self.prepare_mask(occlu_mask1)
        keep_mask2 = self.prepare_mask(occlu_mask2)
        soft_mask1_ = soft_mask1*~occlu_mask1[None,None]
        soft_mask2_ = soft_mask2*~occlu_mask2[None,None]
        soft_mask1 = torch.nn.functional.interpolate(soft_mask1_, (self.opt.ref_size//8, self.opt.ref_size//8), mode="area")
        # soft_mask1 = torch.nn.functional.interpolate(soft_mask1, (self.opt.ref_size//8, self.opt.ref_size//8), mode="area")
        soft_mask1 = soft_mask1.cpu() * (keep_mask1>0.5)
        soft_mask1 = torch.where(obj_mask1 > 0.5, torch.ones_like(soft_mask1), soft_mask1)
        # soft_mask2 = torch.nn.functional.interpolate(soft_mask2, (self.opt.ref_size//8, self.opt.ref_size//8), mode="area")
        soft_mask2 = torch.nn.functional.interpolate(soft_mask2_, (self.opt.ref_size//8, self.opt.ref_size//8), mode="area")
        soft_mask2 = soft_mask2.cpu() * (keep_mask2>0.5)
        soft_mask2 = torch.where(obj_mask2 > 0.5, torch.ones_like(soft_mask2), soft_mask2)
        
        
        # ========== 3. inpaint ==========
        # ddim inversion
        # self.prompt_embeds, self.negative_prompt_embeds = self.ip_adapter.get_prompt_embeds(
        #    self.ref_img, prompt=view_dep_prompt('', 0), negative_prompt="",)
        self.prompt_embeds, self.negative_prompt_embeds = self.ip_adapter.get_prompt_embeds(
           self.ref_img, prompt='', negative_prompt="",)
            
        self.front_start_code, self.front_latents_list, self.front_noise_list = self.ddim_inv(self.input_img_torch, 0)
        self.set_kvcache()
        
        # self.front_start_code2, self.front_latents_list2, self.front_noise_list2 = self.front_start_code.detach().clone(), self.front_latents_list.detach().clone()
        start_code_1, latents_list_1, noise_list_1 = self.ddim_inv(outputs1['image'].permute(2,0,1)[None], self.angles[1])
        start_code_2, latents_list_2, noise_list_2= self.ddim_inv(outputs2['image'].permute(2,0,1)[None], self.angles[2]) 
        # do inpaint
        image1 = self.inpaint(outputs1, soft_mask1, self.angles[1], start_code_1, latents_list_1)
        image2 = self.inpaint(outputs2, soft_mask2, self.angles[2], start_code_2, latents_list_2)
        
        # keep_mask1_ = F.interpolate(keep_mask1, (self.opt.ref_size, self.opt.ref_size), mode="nearest")
        # keep_mask2_ = F.interpolate(keep_mask2, (self.opt.ref_size, self.opt.ref_size), mode="nearest")
        # img = torch.from_numpy(np.array(image1)).permute(2,0,1)[None].float() / 255
        # img = keep_mask1_ * outputs1['image'].permute(2,0,1)[None].cpu() + (1-keep_mask1_) * img
        # image1 = Image.fromarray(img[0].mul(255).permute(1,2,0).numpy().astype(np.uint8))
        # img = torch.from_numpy(np.array(image2)).permute(2,0,1)[None].float() / 255
        # img = keep_mask2_ * outputs2['image'].permute(2,0,1)[None].cpu() + (1-keep_mask2_) * img
        # image2 = Image.fromarray(img[0].mul(255).permute(1,2,0).numpy().astype(np.uint8))
        
        # ========== 4. update known views ==========
        self.save_update_ref(np.array(image1), (outputs1['alpha']>0).cpu().numpy(), f"{self.debug_path}/{self.paint_step}_{self.angles[1]}_rgba.png", self.angles[1])
        self.save_update_ref(np.array(image2), (outputs2['alpha']>0).cpu().numpy(), f"{self.debug_path}/{self.paint_step}_{self.angles[2]}_rgba.png", self.angles[2])

        self.occlu_mask.append((outputs['alpha']>0)[None].permute(0,3,1,2).detach().to(self.device))
        # self.update_prev_occlu_mask(0, soft_mask1.to(self.device), grid1, (outputs1['alpha']>0)[None].permute(0,3,1,2))
        # self.update_prev_occlu_mask(0, soft_mask2.to(self.device), grid2, (outputs2['alpha']>0)[None].permute(0,3,1,2))
        soft_mask1_ = F.interpolate((soft_mask1 < 1).float(), (self.opt.ref_size, self.opt.ref_size), mode="nearest")
        occlu_mask1_ = (soft_mask1_ > 0.5).to(self.device) & (outputs1['alpha']>0)[None].permute(0,3,1,2)
        soft_mask2_ = F.interpolate((soft_mask2 < 1).float(), (self.opt.ref_size, self.opt.ref_size), mode="nearest")
        occlu_mask2_ = (soft_mask2_ > 0.5).to(self.device) & (outputs2['alpha']>0)[None].permute(0,3,1,2)
        occlu_mask_img = Image.fromarray(occlu_mask1_[0,0].to(torch.uint8).mul(255).cpu().numpy())
        occlu_mask_img.save(os.path.join(self.debug_path, f'{self.paint_step}_{self.angles[1]}_occlu_mask_soft.jpg'))
        occlu_mask_img = Image.fromarray(occlu_mask2_[0,0].to(torch.uint8).mul(255).cpu().numpy())
        occlu_mask_img.save(os.path.join(self.debug_path, f'{self.paint_step}_{self.angles[2]}_occlu_mask_soft.jpg'))
        self.occlu_mask.append(occlu_mask1_.to(self.device).detach().clone())
        self.occlu_mask.append(occlu_mask2_.to(self.device).detach().clone())
        occlu_mask_img1.save(os.path.join(self.debug_path, f'{self.paint_step}_{self.angles[1]}_occlu_mask.jpg'))
        occlu_mask_img2.save(os.path.join(self.debug_path, f'{self.paint_step}_{self.angles[2]}_occlu_mask.jpg'))
        keep_mask1_ = F.interpolate(keep_mask1, (self.opt.ref_size, self.opt.ref_size), mode="nearest")
        keep_mask2_ = F.interpolate(keep_mask2, (self.opt.ref_size, self.opt.ref_size), mode="nearest")
        keep_image1 = self.input_img_torch[1:2] * keep_mask1_.to(self.device)
        keep_image2 = self.input_img_torch[2:3] * keep_mask2_.to(self.device)
        Image.fromarray(keep_image1[0].permute(1,2,0).mul(255).to(torch.uint8).cpu().numpy()).save(os.path.join(self.debug_path, f'{self.paint_step}_{self.angles[1]}_keep_image.jpg'))
        Image.fromarray(keep_image2[0].permute(1,2,0).mul(255).to(torch.uint8).cpu().numpy()).save(os.path.join(self.debug_path, f'{self.paint_step}_{self.angles[2]}_keep_image.jpg'))
        
        Image.fromarray(soft_mask1[0,0].mul(255).to(torch.uint8).cpu().numpy()).save(os.path.join(self.debug_path, f'{self.paint_step}_{self.angles[1]}_soft_mask1.jpg'))
        Image.fromarray(soft_mask2[0,0].mul(255).to(torch.uint8).cpu().numpy()).save(os.path.join(self.debug_path, f'{self.paint_step}_{self.angles[2]}_soft_mask2.jpg'))
        Image.fromarray(keep_mask1[0,0].mul(255).to(torch.uint8).cpu().numpy()).save(os.path.join(self.debug_path, f'{self.paint_step}_{self.angles[1]}_keep_mask1.jpg'))
        Image.fromarray(keep_mask2[0,0].mul(255).to(torch.uint8).cpu().numpy()).save(os.path.join(self.debug_path, f'{self.paint_step}_{self.angles[2]}_keep_mask2.jpg'))
        Image.fromarray(outputs1['image'].mul(255).to(torch.uint8).cpu().numpy()).save(os.path.join(self.debug_path, f'{self.paint_step}_{self.angles[1]}_input_img.jpg'))
        Image.fromarray(outputs2['image'].mul(255).to(torch.uint8).cpu().numpy()).save(os.path.join(self.debug_path, f'{self.paint_step}_{self.angles[2]}_input_img.jpg'))
        
        occlu_masks = [m.float() for m in self.occlu_mask]
        occlu_imgs = [self.occlu_mask[i].float() * self.input_img_torch[i] for i in range(len(self.occlu_mask))]
        save_tensor2image(torch.cat(occlu_masks), os.path.join(self.debug_path, f'{self.paint_step}_occlu_masks.jpg'))
        save_tensor2image(torch.cat(occlu_imgs), os.path.join(self.debug_path, f'{self.paint_step}_occlu_imgs.jpg'))
        
        
    @torch.no_grad()
    def paint_bidirectional(self):
        # ========== 1. render views ==========
        outputs1 = self.render_view(self.opt.elevation, self.angles[self.paint_step-2])
        outputs2 = self.render_view(self.opt.elevation, self.angles[self.paint_step-1])
        outputs = self.render_view(self.opt.elevation, self.angles[self.paint_step])
        
        # ========== 2. caculate inpainting masks ==========
        occlu_mask1, grid1, surface_pts_masked1, ndc_coords1, soft_mask1 = self.backward_oculusion_aware_render(
            outputs, outputs1, 0, self.angles[self.paint_step]-self.angles[self.paint_step-2], rm_edge=True)
        occlu_mask2, grid2, surface_pts_masked2, ndc_coords2, soft_mask2 = self.backward_oculusion_aware_render(
            outputs, outputs2, 0, self.angles[self.paint_step]-self.angles[self.paint_step-1], rm_edge=True)
        occlu_mask = occlu_mask1 & occlu_mask2
        occlu_mask_img = Image.fromarray(occlu_mask.to(torch.uint8).mul(255).cpu().numpy())
        keep_mask1, keep_mask2 = self.prepare_mask(occlu_mask1), self.prepare_mask(occlu_mask2)
        soft_mask1_ = soft_mask1*~occlu_mask1[None,None]
        soft_mask2_ = soft_mask2*~occlu_mask2[None,None]
        soft_mask1 = torch.nn.functional.interpolate(soft_mask1_, (self.opt.ref_size//8, self.opt.ref_size//8), mode="area")
        soft_mask2 = torch.nn.functional.interpolate(soft_mask2_, (self.opt.ref_size//8, self.opt.ref_size//8), mode="area")
        soft_mask1 = soft_mask1.cpu() * (keep_mask1>0.5)
        soft_mask2 = soft_mask2.cpu() * (keep_mask2>0.5)
        obj_mask = self.prepare_mask(outputs["alpha"][...,0]>0)
        soft_mask1 = torch.where(obj_mask > 0.5, torch.ones_like(soft_mask1), soft_mask1)
        soft_mask2 = torch.where(obj_mask > 0.5, torch.ones_like(soft_mask2), soft_mask2)
        soft_mask = torch.max(soft_mask1, soft_mask2)
        keep_mask = self.prepare_mask(occlu_mask)
        soft_mask = soft_mask.cpu() * (keep_mask>0.5)
        soft_mask = torch.where(obj_mask > 0.5, torch.ones_like(soft_mask), soft_mask)
        
        # ========== 3. inpaint ==========
        # ddim inversion
        start_code, latents_list, noise_list = self.ddim_inv(outputs['image'].permute(2,0,1)[None], self.angles[self.paint_step])
        # do inpaint
        image = self.inpaint(outputs, soft_mask, self.angles[self.paint_step], start_code, latents_list)
        
        # keep_mask_ = F.interpolate(keep_mask, (self.opt.ref_size, self.opt.ref_size), mode="nearest")
        # img = torch.from_numpy(np.array(image)).permute(2,0,1)[None].float() / 255
        # img = keep_mask_ * outputs['image'].permute(2,0,1)[None].cpu() + (1-keep_mask_) * img
        # image = Image.fromarray(img[0].mul(255).permute(1,2,0).numpy().astype(np.uint8))
        
        # ========== 4. update known views ==========
        self.save_update_ref(np.array(image), (outputs['alpha']>0).cpu().numpy(), f"{self.debug_path}/{self.paint_step}_{self.angles[self.paint_step]}_rgba.png", self.angles[self.paint_step])


        Image.fromarray(outputs['image'].mul(255).to(torch.uint8).cpu().numpy()).save(os.path.join(self.debug_path, f'{self.paint_step}_{self.angles[self.paint_step]}_input_img.jpg'))
        Image.fromarray(keep_mask[0,0].mul(255).to(torch.uint8).cpu().numpy()).save(os.path.join(self.debug_path, f'{self.paint_step}_{self.angles[self.paint_step]}_keep_mask.jpg'))
        Image.fromarray(soft_mask[0,0].mul(255).to(torch.uint8).cpu().numpy()).save(os.path.join(self.debug_path, f'{self.paint_step}_{self.angles[self.paint_step]}_soft_mask.jpg'))
        soft_mask_ = F.interpolate((soft_mask < 1).float(), (self.opt.ref_size, self.opt.ref_size), mode="nearest")
        occlu_mask_ = (soft_mask_ > 0.5).to(self.device) & (outputs['alpha']>0)[None].permute(0,3,1,2)
        self.occlu_mask.append(occlu_mask_.to(self.device).detach().clone())
        # self.update_prev_occlu_mask(self.paint_step-2, soft_mask1.to(self.device), grid1, (outputs['alpha']>0)[None].permute(0,3,1,2))
        # self.update_prev_occlu_mask(self.paint_step-1, soft_mask2.to(self.device), grid2, (outputs['alpha']>0)[None].permute(0,3,1,2))
        occlu_mask_img = Image.fromarray(occlu_mask_[0,0].to(torch.uint8).mul(255).cpu().numpy())
        occlu_mask_img.save(os.path.join(self.debug_path, f'{self.paint_step}_{self.angles[self.paint_step]}_occlu_mask_soft.jpg'))
        occlu_mask_img1 = Image.fromarray(occlu_mask1.to(torch.uint8).mul(255).cpu().numpy())
        occlu_mask_img2 = Image.fromarray(occlu_mask2.to(torch.uint8).mul(255).cpu().numpy())
        occlu_mask_img.save(os.path.join(self.debug_path, f'{self.paint_step}_{self.angles[self.paint_step]}_occlu_mask.jpg'))
        occlu_mask_img1.save(os.path.join(self.debug_path, f'{self.paint_step}_{self.angles[self.paint_step]}_occlu_mask1.jpg'))
        occlu_mask_img2.save(os.path.join(self.debug_path, f'{self.paint_step}_{self.angles[self.paint_step]}_occlu_mask2.jpg'))
        # assert torch.all(self.front_start_code2 == self.front_start_code[:1])
        # assert torch.all(self.front_latents_list2 == self.front_latents_list[:1])
        keep_mask_ = F.interpolate(keep_mask, (self.opt.ref_size, self.opt.ref_size), mode="nearest")
        keep_image = self.input_img_torch[self.paint_step:self.paint_step+1] * keep_mask_.to(self.device)
        Image.fromarray(keep_image[0].permute(1,2,0).mul(255).to(torch.uint8).cpu().numpy()).save(os.path.join(self.debug_path, f'{self.paint_step}_{self.angles[self.paint_step]}_keep_image.jpg'))
        occlu_masks = [m.float() for m in self.occlu_mask]
        save_tensor2image(torch.cat(occlu_masks), os.path.join(self.debug_path, f'{self.paint_step}_occlu_masks.jpg'))
        occlu_imgs = [self.occlu_mask[i].float() * self.input_img_torch[i] for i in range(len(self.occlu_mask))]
        save_tensor2image(torch.cat(occlu_imgs), os.path.join(self.debug_path, f'{self.paint_step}_occlu_imgs.jpg'))
        
    def train_step(self):
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()


        for _ in range(self.train_steps):

            self.step += 1
            step_ratio = min(1, self.step / self.opt.iters_refine)
            # self.denoising_strength = (1 - step_ratio)* 0.45 + 0.5 # from 0.95 to 0.5

            loss = 0

            if self.input_img_torch.shape[0] == 1 or not self.opt.iterative:
                loss_choice = 0
            elif self.input_img_torch.shape[0] <= 3:
                loss_choice = torch.randperm(2)[0]
            else:
                loss_choice = torch.randperm(3)[0]
            
            # loss_choice = 0
            if loss_choice == 0:
                ### known view
                if self.input_img_torch is not None:
                    
                    self.input_img_torch_channel_last = self.input_img_torch[0].permute(1,2,0).contiguous()
                    
                    ssaa = min(2.0, max(0.125, 2 * np.random.random()))
                    out = self.renderer.render(*self.fixed_cam[0], self.opt.ref_size, self.opt.ref_size, ssaa=ssaa)

                    # rgb loss
                    image = out["image"] # [H, W, 3] in [0, 1]
                    # valid_mask = ((out["alpha"] > 0) & (out["viewcos"] > 0.5)).detach()
                    valid_mask = (out["alpha"] > 0).detach()
                    valid_mask = torch.from_numpy(self.rm_edge(valid_mask.cpu().numpy())).to(self.input_img_torch_channel_last.device)
                    loss = loss + F.mse_loss(image * valid_mask, self.input_img_torch_channel_last * valid_mask)
            
            # loss_choice = 1
            if loss_choice == 1:
                ### novel view
                if self.opt.iterative and self.input_img_torch.shape[0] > 1:
                    
                    choice = torch.randperm(self.input_img_torch.shape[0]-1)[0] + 1
                    self.input_img_torch_channel_last = self.input_img_torch[choice].permute(1,2,0).contiguous()
                    
                    ssaa = min(2.0, max(0.125, 2 * np.random.random()))
                    out = self.renderer.render(*self.fixed_cam[choice], self.opt.ref_size, self.opt.ref_size, ssaa=ssaa)

                    # rgb loss
                    image = out["image"] # [H, W, 3] in [0, 1]
                    valid_mask = ((out["alpha"] > 0)).detach()
                    valid_mask = torch.from_numpy(self.rm_edge(valid_mask.cpu().numpy())).to(self.input_img_torch_channel_last.device)
                    # loss = loss + F.mse_loss(image * valid_mask, self.input_img_torch_channel_last * valid_mask)
                    loss = loss + F.mse_loss(image * valid_mask*self.occlu_mask[choice][...,None], self.input_img_torch_channel_last * valid_mask*self.occlu_mask[choice][...,None])
            
            # loss_choice = 2
            if loss_choice == 2:
                ### last view
                if self.opt.iterative and self.input_img_torch.shape[0] > 3:
                    
                    choice = -1
                    self.input_img_torch_channel_last = self.input_img_torch[choice].permute(1,2,0).contiguous()
                    
                    ssaa = min(2.0, max(0.125, 2 * np.random.random()))
                    out = self.renderer.render(*self.fixed_cam[choice], self.opt.ref_size, self.opt.ref_size, ssaa=ssaa)

                    # rgb loss
                    image = out["image"] # [H, W, 3] in [0, 1]
                    valid_mask = ((out["alpha"] > 0)).detach()
                    valid_mask = torch.from_numpy(self.rm_edge(valid_mask.cpu().numpy())).to(self.input_img_torch_channel_last.device)
                    # loss = loss + F.mse_loss(image * valid_mask, self.input_img_torch_channel_last * valid_mask)
                    loss = loss + F.mse_loss(image * valid_mask*self.occlu_mask[choice][...,None], self.input_img_torch_channel_last * valid_mask*self.occlu_mask[choice][...,None])

            ### novel view (manual batch)
            render_resolution = 512
            images = []
            poses = []
            vers, hors, radii = [], [], []
            # avoid too large elevation (> 80 or < -80), and make sure it always cover [-30, 30]
            min_ver = max(min(-30, -30 - self.opt.elevation), -80 - self.opt.elevation)
            max_ver = min(max(30, 30 - self.opt.elevation), 80 - self.opt.elevation)
            for _ in range(self.opt.batch_size):

                # render random view
                ver = np.random.randint(min_ver, max_ver)
                hor = np.random.randint(-180, 180)
                radius = 0

                vers.append(ver)
                hors.append(hor)
                radii.append(radius)

                pose = orbit_camera(self.opt.elevation + ver, hor, self.opt.radius + radius)
                # pose = orbit_camera(self.opt.elevation, 40, self.opt.radius)
                poses.append(pose)

                # random render resolution
                ssaa = min(2.0, max(0.125, 2 * np.random.random()))
                out = self.renderer.render(pose, self.cam.perspective, render_resolution, render_resolution, ssaa=ssaa)

                image = out["image"] # [H, W, 3] in [0, 1]
                image = image.permute(2,0,1).contiguous().unsqueeze(0) # [1, 3, H, W] in [0, 1]

                images.append(image)

                # enable mvdream training
                if self.opt.mvdream:
                    for view_i in range(1, 4):
                        pose_i = orbit_camera(self.opt.elevation + ver, hor + 90 * view_i, self.opt.radius + radius)
                        poses.append(pose_i)

                        out_i = self.renderer.render(pose_i, self.cam.perspective, render_resolution, render_resolution, ssaa=ssaa)

                        image = out_i["image"].permute(2,0,1).contiguous().unsqueeze(0) # [1, 3, H, W] in [0, 1]
                        images.append(image)

            images = torch.cat(images, dim=0)
            poses = torch.from_numpy(np.stack(poses, axis=0)).to(self.device)

            # import kiui
            # kiui.lo(hor, ver)
            # kiui.vis.plot_image(image)

            # guidance loss
            strength = step_ratio * 0.45 + 0.5 # from 0.5 to 0.95
            if self.enable_sd and not self.opt.iterative:
                if self.opt.mvdream:
                    # loss = loss + self.opt.lambda_sd * self.guidance_sd.train_step(images, poses, step_ratio)
                    refined_images = self.guidance_sd.refine(images, poses, strength=strength).float()
                    refined_images = F.interpolate(refined_images, (render_resolution, render_resolution), mode="bilinear", align_corners=False)
                    loss = loss + self.opt.lambda_sd * F.mse_loss(images, refined_images)
                else:
                    # loss = loss + self.opt.lambda_sd * self.guidance_sd.train_step(images, step_ratio)
                    refined_images = self.guidance_sd.refine(images, strength=strength).float()
                    refined_images = F.interpolate(refined_images, (render_resolution, render_resolution), mode="bilinear", align_corners=False)
                    loss = loss + self.opt.lambda_sd * F.mse_loss(images, refined_images)

            if self.enable_zero123 and not self.opt.iterative:
                # loss = loss + self.opt.lambda_zero123 * self.guidance_zero123.train_step(images, vers, hors, radii, step_ratio)
                refined_images = self.guidance_zero123.refine(images, vers, hors, radii, strength=strength).float()
                refined_images = F.interpolate(refined_images, (render_resolution, render_resolution), mode="bilinear", align_corners=False)
                loss = loss + self.opt.lambda_zero123 * F.mse_loss(images, refined_images)
                # loss = loss + self.opt.lambda_zero123 * self.lpips_loss(images, refined_images)

            # optimize step
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            if self.opt.iterative and self.step // self.interval > ((self.step-1) // self.interval) and self.paint_step < self.num_views-1:
                self.paint_step = max(0, int(self.step // self.interval) - 1)
                if self.paint_step == 1:
                    self.paint_facade()
                elif self.paint_step >= 3:
                    self.paint_bidirectional()
                self.optimizer.zero_grad()

        ender.record()
        torch.cuda.synchronize()
        t = starter.elapsed_time(ender)

        self.need_update = True

        if self.gui:
            dpg.set_value("_log_train_time", f"{t:.4f}ms")
            dpg.set_value(
                "_log_train_log",
                f"step = {self.step: 5d} (+{self.train_steps: 2d}) loss = {loss.item():.4f}",
            )

        # dynamic train steps (no need for now)
        # max allowed train time per-frame is 500 ms
        # full_t = t / self.train_steps * 16
        # train_steps = min(16, max(4, int(16 * 500 / full_t)))
        # if train_steps > self.train_steps * 1.2 or train_steps < self.train_steps * 0.8:
        #     self.train_steps = train_steps

    @torch.no_grad()
    def test_step(self):
        # ignore if no need to update
        if not self.need_update:
            return

        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()

        # should update image
        if self.need_update:
            # render image

            out = self.renderer.render(self.cam.pose, self.cam.perspective, self.H, self.W)

            buffer_image = out[self.mode]  # [H, W, 3]

            if self.mode in ['depth', 'alpha']:
                buffer_image = buffer_image.repeat(1, 1, 3)
                if self.mode == 'depth':
                    buffer_image = (buffer_image - buffer_image.min()) / (buffer_image.max() - buffer_image.min() + 1e-20)

            self.buffer_image = buffer_image.contiguous().clamp(0, 1).detach().cpu().numpy()
            
            # display input_image
            if self.overlay_input_img and self.input_img is not None:
                self.buffer_image = (
                    self.buffer_image * (1 - self.overlay_input_img_ratio)
                    + self.input_img * self.overlay_input_img_ratio
                )

            self.need_update = False

        ender.record()
        torch.cuda.synchronize()
        t = starter.elapsed_time(ender)

        if self.gui:
            dpg.set_value("_log_infer_time", f"{t:.4f}ms ({int(1000/t)} FPS)")
            dpg.set_value(
                "_texture", self.buffer_image
            )  # buffer must be contiguous, else seg fault!

    def rm_edge(self, mask, edge_width=3):
        alpha = np.uint8(mask.astype(np.float32)*255)
        erode = cv2.erode(alpha, np.ones((edge_width, edge_width), np.uint8))
        edge = cv2.absdiff(alpha, erode).astype(np.float32) / 255
        mask_no_edge = mask > 0.5
        mask_no_edge[edge>0.1] = False
        return mask_no_edge
    
    def load_input(self, file):
        # load image
        print(f'[INFO] load image from {file}...')
        img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
        if img.shape[-1] == 3:
            if self.bg_remover is None:
                self.bg_remover = rembg.new_session()
            img = rembg.remove(img, session=self.bg_remover)
   
        img = cv2.resize(
            img, (self.W, self.H), interpolation=cv2.INTER_AREA
        )
        img = img.astype(np.float32) / 255.0

        self.input_mask = img[..., 3:]
        # white bg
        self.input_img = img[..., :3] * self.input_mask + (
            1 - self.input_mask
        )
        # bgr to rgb
        self.input_img = self.input_img[..., ::-1].copy()

        # load prompt
        file_prompt = file.replace("_rgba.png", "_caption.txt")
        if os.path.exists(file_prompt):
            print(f'[INFO] load prompt from {file_prompt}...')
            with open(file_prompt, "r") as f:
                self.prompt = f.read().strip()
    
    def save_model(self):
        os.makedirs(self.opt.outdir, exist_ok=True)
    
        path = os.path.join(self.opt.outdir, self.opt.save_path + '.' + self.opt.mesh_format)
        self.renderer.export_mesh(path)

        print(f"[INFO] save model to {path}.")

    def register_dpg(self):
        ### register texture

        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(
                self.W,
                self.H,
                self.buffer_image,
                format=dpg.mvFormat_Float_rgb,
                tag="_texture",
            )

        ### register window

        # the rendered image, as the primary window
        with dpg.window(
            tag="_primary_window",
            width=self.W,
            height=self.H,
            pos=[0, 0],
            no_move=True,
            no_title_bar=True,
            no_scrollbar=True,
        ):
            # add the texture
            dpg.add_image("_texture")

        # dpg.set_primary_window("_primary_window", True)

        # control window
        with dpg.window(
            label="Control",
            tag="_control_window",
            width=600,
            height=self.H,
            pos=[self.W, 0],
            no_move=True,
            no_title_bar=True,
        ):
            # button theme
            with dpg.theme() as theme_button:
                with dpg.theme_component(dpg.mvButton):
                    dpg.add_theme_color(dpg.mvThemeCol_Button, (23, 3, 18))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (51, 3, 47))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (83, 18, 83))
                    dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5)
                    dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 3, 3)

            # timer stuff
            with dpg.group(horizontal=True):
                dpg.add_text("Infer time: ")
                dpg.add_text("no data", tag="_log_infer_time")

            def callback_setattr(sender, app_data, user_data):
                setattr(self, user_data, app_data)

            # init stuff
            with dpg.collapsing_header(label="Initialize", default_open=True):

                # seed stuff
                def callback_set_seed(sender, app_data):
                    self.seed = app_data
                    self.seed_everything()

                dpg.add_input_text(
                    label="seed",
                    default_value=self.seed,
                    on_enter=True,
                    callback=callback_set_seed,
                )

                # input stuff
                def callback_select_input(sender, app_data):
                    # only one item
                    for k, v in app_data["selections"].items():
                        dpg.set_value("_log_input", k)
                        self.load_input(v)

                    self.need_update = True

                with dpg.file_dialog(
                    directory_selector=False,
                    show=False,
                    callback=callback_select_input,
                    file_count=1,
                    tag="file_dialog_tag",
                    width=700,
                    height=400,
                ):
                    dpg.add_file_extension("Images{.jpg,.jpeg,.png}")

                with dpg.group(horizontal=True):
                    dpg.add_button(
                        label="input",
                        callback=lambda: dpg.show_item("file_dialog_tag"),
                    )
                    dpg.add_text("", tag="_log_input")
                
                # overlay stuff
                with dpg.group(horizontal=True):

                    def callback_toggle_overlay_input_img(sender, app_data):
                        self.overlay_input_img = not self.overlay_input_img
                        self.need_update = True

                    dpg.add_checkbox(
                        label="overlay image",
                        default_value=self.overlay_input_img,
                        callback=callback_toggle_overlay_input_img,
                    )

                    def callback_set_overlay_input_img_ratio(sender, app_data):
                        self.overlay_input_img_ratio = app_data
                        self.need_update = True

                    dpg.add_slider_float(
                        label="ratio",
                        min_value=0,
                        max_value=1,
                        format="%.1f",
                        default_value=self.overlay_input_img_ratio,
                        callback=callback_set_overlay_input_img_ratio,
                    )

                # prompt stuff
            
                dpg.add_input_text(
                    label="prompt",
                    default_value=self.prompt,
                    callback=callback_setattr,
                    user_data="prompt",
                )

                dpg.add_input_text(
                    label="negative",
                    default_value=self.negative_prompt,
                    callback=callback_setattr,
                    user_data="negative_prompt",
                )

                # save current model
                with dpg.group(horizontal=True):
                    dpg.add_text("Save: ")

                    dpg.add_button(
                        label="model",
                        tag="_button_save_model",
                        callback=self.save_model,
                    )
                    dpg.bind_item_theme("_button_save_model", theme_button)

                    dpg.add_input_text(
                        label="",
                        default_value=self.opt.save_path,
                        callback=callback_setattr,
                        user_data="save_path",
                    )

            # training stuff
            with dpg.collapsing_header(label="Train", default_open=True):
                # lr and train button
                with dpg.group(horizontal=True):
                    dpg.add_text("Train: ")

                    def callback_train(sender, app_data):
                        if self.training:
                            self.training = False
                            dpg.configure_item("_button_train", label="start")
                        else:
                            self.prepare_train()
                            self.training = True
                            dpg.configure_item("_button_train", label="stop")

                    # dpg.add_button(
                    #     label="init", tag="_button_init", callback=self.prepare_train
                    # )
                    # dpg.bind_item_theme("_button_init", theme_button)

                    dpg.add_button(
                        label="start", tag="_button_train", callback=callback_train
                    )
                    dpg.bind_item_theme("_button_train", theme_button)

                with dpg.group(horizontal=True):
                    dpg.add_text("", tag="_log_train_time")
                    dpg.add_text("", tag="_log_train_log")

            # rendering options
            with dpg.collapsing_header(label="Rendering", default_open=True):
                # mode combo
                def callback_change_mode(sender, app_data):
                    self.mode = app_data
                    self.need_update = True

                dpg.add_combo(
                    ("image", "depth", "alpha", "normal"),
                    label="mode",
                    default_value=self.mode,
                    callback=callback_change_mode,
                )

                # fov slider
                def callback_set_fovy(sender, app_data):
                    self.cam.fovy = np.deg2rad(app_data)
                    self.need_update = True

                dpg.add_slider_int(
                    label="FoV (vertical)",
                    min_value=1,
                    max_value=120,
                    format="%d deg",
                    default_value=np.rad2deg(self.cam.fovy),
                    callback=callback_set_fovy,
                )

        ### register camera handler

        def callback_camera_drag_rotate_or_draw_mask(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            dx = app_data[1]
            dy = app_data[2]

            self.cam.orbit(dx, dy)
            self.need_update = True

        def callback_camera_wheel_scale(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            delta = app_data

            self.cam.scale(delta)
            self.need_update = True

        def callback_camera_drag_pan(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            dx = app_data[1]
            dy = app_data[2]

            self.cam.pan(dx, dy)
            self.need_update = True

        def callback_set_mouse_loc(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            # just the pixel coordinate in image
            self.mouse_loc = np.array(app_data)

        with dpg.handler_registry():
            # for camera moving
            dpg.add_mouse_drag_handler(
                button=dpg.mvMouseButton_Left,
                callback=callback_camera_drag_rotate_or_draw_mask,
            )
            dpg.add_mouse_wheel_handler(callback=callback_camera_wheel_scale)
            dpg.add_mouse_drag_handler(
                button=dpg.mvMouseButton_Middle, callback=callback_camera_drag_pan
            )

        dpg.create_viewport(
            title="Gaussian3D",
            width=self.W + 600,
            height=self.H + (45 if os.name == "nt" else 0),
            resizable=False,
        )

        ### global theme
        with dpg.theme() as theme_no_padding:
            with dpg.theme_component(dpg.mvAll):
                # set all padding to 0 to avoid scroll bar
                dpg.add_theme_style(
                    dpg.mvStyleVar_WindowPadding, 0, 0, category=dpg.mvThemeCat_Core
                )
                dpg.add_theme_style(
                    dpg.mvStyleVar_FramePadding, 0, 0, category=dpg.mvThemeCat_Core
                )
                dpg.add_theme_style(
                    dpg.mvStyleVar_CellPadding, 0, 0, category=dpg.mvThemeCat_Core
                )

        dpg.bind_item_theme("_primary_window", theme_no_padding)

        dpg.setup_dearpygui()

        ### register a larger font
        # get it from: https://github.com/lxgw/LxgwWenKai/releases/download/v1.300/LXGWWenKai-Regular.ttf
        if os.path.exists("LXGWWenKai-Regular.ttf"):
            with dpg.font_registry():
                with dpg.font("LXGWWenKai-Regular.ttf", 18) as default_font:
                    dpg.bind_font(default_font)

        # dpg.show_metrics()

        dpg.show_viewport()

    def render(self):
        assert self.gui
        while dpg.is_dearpygui_running():
            # update texture every frame
            if self.training:
                self.train_step()
            self.test_step()
            dpg.render_dearpygui_frame()
    
    # no gui mode
    def train(self, iters=500):
        if iters > 0:
            self.prepare_train()
            for i in tqdm.trange(iters):
                self.train_step()
        # save
        self.save_model()
        

if __name__ == "__main__":
    import argparse
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to the yaml config file")
    parser.add_argument("--ddnm_strength",type=float, default=0.4, help="")
    args, extras = parser.parse_known_args()

    # override default config from cli
    opt = OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_cli(extras))
    opt.ddnm_strength = args.ddnm_strength
    
    # auto find mesh from stage 1
    if opt.mesh is None:
        default_path = os.path.join(opt.outdir, opt.save_path + '_mesh.' + opt.mesh_format)
        if os.path.exists(default_path):
            opt.mesh = default_path
        else:
            raise ValueError(f"Cannot find mesh from {default_path}, must specify --mesh explicitly!")

    gui = GUI(opt)

    if opt.gui:
        gui.render()
    else:
        gui.train(opt.iters_refine)
