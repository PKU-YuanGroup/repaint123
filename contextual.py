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
from contextual_loss_pytorch.contextual_loss import *
from torchvision import transforms

# import debugpy; debugpy.connect(('localhost', 5677))

class Contextual:

    def __init__(self,
                 device=None,
                 #clip_name='models/models--laion--CLIP-ViT-bigG-14-laion2B-39B-b160k/snapshots/8c7a3583335de4dba1b07182dbf81c75137ce67b',
                 size=192):  #'laion/CLIP-ViT-B-32-laion2B-s34B-b79K'):
        super().__init__()
        self.size = size
        self.device = f"cuda:{device}"
        # self.device = device if device is not None else torch.device(
        #     'cuda' if torch.cuda.is_available() else 'cpu')
        #clip_name = clip_name

        # self.feature_extractor = CLIPFeatureExtractor.from_pretrained(
        #     clip_name)
        # self.clip_model = CLIPModel.from_pretrained(clip_name).to(self.device)
        # self.tokenizer = CLIPTokenizer.from_pretrained(
        #     'models/models--openai--clip-vit-base-patch32/snapshots/e6a30b603a447e251fdaca1c3056b2a16cdfebeb')

        # self.normalize = transforms.Normalize(
        #     mean=self.feature_extractor.image_mean,
        #     std=self.feature_extractor.image_std)

        self.resize = transforms.Resize(192)
        self.to_tensor = transforms.ToTensor()

        # image augmentation
        self.aug = T.Compose([
            T.Resize((192, 192)),
            T.Normalize((0.48145466, 0.4578275, 0.40821073),
                        (0.26862954, 0.26130258, 0.27577711)),
        ])
        self.criterion = ContextualLoss(use_vgg=True, vgg_layer='relu5_4').cuda(device=self.device)
    # * recommend to use this function for evaluation
    @torch.no_grad()
    def score_gt(self, ref_img_path, novel_views):
        # assert len(novel_views) == 100
        contextual_scores = []
        for novel in novel_views:
            contextual_scores.append(self.score_from_path(ref_img_path, [novel]).cpu().detach())
        return np.mean(contextual_scores)

    # * recommend to use this function for evaluation
    # def score_gt(self, ref_paths, novel_paths):
    #     clip_scores = []
    #     for img1_path, img2_path in zip(ref_paths, novel_paths):
    #         clip_scores.append(self.score_from_path(img1_path, img2_path))

    #     return np.mean(clip_scores)

    def similarity(self, image1_features: torch.Tensor,
                   image2_features: torch.Tensor) -> float:
        with torch.no_grad(), torch.cuda.amp.autocast():
            y = image1_features.T.view(image1_features.T.shape[1],
                                       image1_features.T.shape[0])
            similarity = torch.matmul(y, image2_features.T)
            # print(similarity)
            return similarity[0][0].item()

    def get_img_embeds(self, img):
        if img.shape[0] == 4:
            img = img[:3, :, :]

        # img = self.aug(img).to(self.device)
        img = img.unsqueeze(0).to(self.device) # b,c,h,w

        # plt.imshow(img.cpu().squeeze(0).permute(1, 2, 0).numpy())
        # plt.show()
        # print(img)

        # image_z = self.clip_model.get_image_features(img)
        # image_z = image_z / image_z.norm(dim=-1,
        #                                  keepdim=True)  # normalize features
        return img

    def score_from_feature(self, img1, img2):
        img1_feature, img2_feature = self.get_img_embeds(
            img1), self.get_img_embeds(img2)
        
        # for debug
        return self.criterion(img1_feature, img2_feature)

    def read_img_list(self, img_list):
        size = self.size
        images = []
        # white_background = np.ones((size, size, 3), dtype=np.uint8) * 255

        for img_path in img_list:
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            # print(img_path)
            if img.shape[2] == 4:  # Handle BGRA images
                alpha = img[:, :, 3]  # Extract alpha channel
                img = cv2.cvtColor(img,cv2.COLOR_BGRA2RGB)  # Convert BGRA to BGR
                img[np.where(alpha == 0)] = [
                    255, 255, 255
                ]  # Set transparent pixels to white
            else:  # Handle other image formats like JPG and PNG
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)

            # plt.imshow(img)
            # plt.show()

            images.append(img)

        images = np.stack(images, axis=0)
        # images[np.where(images == 0)] = 255  # Set black pixels to white
        # images = np.where(images == 0, white_background, images)  # Set transparent pixels to white
        # images = images.astype(np.float32)

        return images

    def score_from_path(self, img1_path, img2_path):
        img1, img2 = self.read_img_list(img1_path), self.read_img_list(img2_path)
        img1 = np.squeeze(img1)
        img2 = np.squeeze(img2)
        # plt.imshow(img1)
        # plt.show()
        # plt.imshow(img2)
        # plt.show()

        img1, img2 = self.to_tensor(img1), self.to_tensor(img2)
        # print("img1 to tensor ",img1)
        return self.score_from_feature(img1, img2)



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
    parser.add_argument('--num_azimuth', type=int, default=100, help="number of images to render from different azimuths")
    parser.add_argument('--force_cuda_rast', action="store_true", help="force cuda rasterizer")
    parser.add_argument('--train_geo', action="store_true", help="")
    opt = parser.parse_args()
    opt.wogui = True

    # clip = CLIP('cuda')
    contextual_scorer = Contextual(0)
    
    gui = GUI(opt)
    gui.renderer = Renderer(opt)

    # load image and encode as ref features
    ref_img = kiui.read_image(opt.image, mode='float')
    if ref_img.shape[-1] == 4:
        # rgba to white-bg rgb
        ref_img = ref_img[..., :3] * ref_img[..., 3:] + (1 - ref_img[..., 3:])
    ref_img = (ref_img * 255).astype(np.uint8)
    ref_img = cv2.resize(ref_img, (opt.H, opt.W), interpolation=cv2.INTER_AREA)
    # ref_img = cv2.resize(ref_img, (192, 192), interpolation=cv2.INTER_AREA)
    ref_img_tensor = torch.from_numpy(ref_img).float().div(255).permute(2,0,1)
    # with torch.no_grad():
    #     ref_features = contextual_scorer.encode_image(ref_img)

    # render from random views and evaluate similarity
    results = []

    elevation = [opt.elevation,]
    azimuth = np.linspace(0, 360, opt.num_azimuth, dtype=np.int32, endpoint=False)
    for ele in tqdm.tqdm(elevation):
        for azi in tqdm.tqdm(azimuth):
            gui.cam.from_angle(ele, azi)
            gui.need_update = True
            gui.step()
            
            # ssaa = min(2.0, max(0.125, 2 * np.random.random()))
            # out = gui.renderer.render(orbit_camera(ele, azi, opt.radius), gui.cam.perspective, opt.H, opt.W, ssaa=ssaa)
            out = gui.renderer.render(orbit_camera(ele, azi, opt.radius), gui.cam.perspective, opt.H, opt.W)
            image_tensor = out["image"].detach().permute(2,0,1)
            
            # image_tensor = out["image"].detach().cpu().numpy() * 255
            # image_tensor = cv2.resize(image_tensor.astype(np.uint8), (192, 192), interpolation=cv2.INTER_AREA)
            # image_tensor = torch.from_numpy(image_tensor).float().div(255).permute(2,0,1)
            
            
            # image = (gui.render_buffer * 255).astype(np.uint8)
            
            # with torch.no_grad():
            #     cur_features = clip.encode_image(image)
            
            # # kiui.lo(ref_features, cur_features)
            # similarity = (ref_features * cur_features).sum(dim=-1).mean().item()

            # results.append(similarity)
            # print(ref_img_tensor.shape, image_tensor.shape)
            with torch.no_grad():
                results.append(contextual_scorer.score_from_feature(image_tensor, ref_img_tensor).cpu().detach())
    
    avg_similarity = np.mean(results)
    print("contextual-distance: ", avg_similarity)

            