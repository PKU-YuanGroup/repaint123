# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import contextlib
import inspect
from typing import Callable, List, Optional, Union, Dict, Any

import torch

import PIL
from diffusers.utils import is_accelerate_available
from packaging import version
from tqdm import tqdm
from transformers import (
    CLIPTextModel,
    CLIPTokenizer,
    DPTFeatureExtractor,
    DPTForDepthEstimation,
)

from diffusers.configuration_utils import FrozenDict
from diffusers.models import AutoencoderKL, UNet2DConditionModel,  ControlNetModel
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from diffusers.schedulers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.utils import PIL_INTERPOLATION, deprecate, logging
from diffusers.utils.torch_utils import is_compiled_module
from diffusers.pipelines import StableDiffusionControlNetPipeline, StableDiffusionControlNetImg2ImgPipeline, StableDiffusionControlNetInpaintPipeline
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

from .mask_options import *

def adain(feat, cond_feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size() # [d, c]
    C = size[1]
    feat_var = feat.var(dim=0) + eps
    feat_std = feat_var.sqrt().view(1, C)
    feat_mean = feat.mean(dim=0).view(1, C)
    
    cond_feat_var = cond_feat.var(dim=0) + eps
    cond_feat_std = cond_feat_var.sqrt().view(1, C)
    cond_feat_mean = cond_feat.mean(dim=0).view(1, C)
    feat = (feat - feat_mean.expand(size)) / feat_std.expand(size)
    return feat * cond_feat_std.expand(size) + cond_feat_mean.expand(size)

def adain_v(feat, cond_feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size() # [d, c]
    C = size[1]
    feat_var = feat.var(dim=0) + eps
    feat_std = feat_var.sqrt().view(1, C)
    feat_mean = feat.mean(dim=0).view(1, C)
    
    cond_feat_var = cond_feat.var(dim=0) + eps
    cond_feat_std = cond_feat_var.sqrt().view(1, C)
    feat = (feat - feat_mean.expand(size)) / feat_std.expand(size)
    return feat * cond_feat_std.expand(size) + feat_mean.expand(size)

def masked_adain(feat, cond_feat, mask, cond_mask, eps=1e-5):
    mask = mask.expand_as(feat)
    cond_mask = cond_mask.expand_as(cond_feat)
    feat[mask] = adain(feat[mask].reshape(4,-1).transpose(0,1), cond_feat[cond_mask].reshape(4,-1).transpose(0,1), eps).transpose(0,1).flatten()
    feat[~mask] = adain(feat[~mask].reshape(4,-1).transpose(0,1), cond_feat[~cond_mask].reshape(4,-1).transpose(0,1), eps).transpose(0,1).flatten()
    return feat

def masked_adain_v(feat, cond_feat, mask, cond_mask, eps=1e-5):
    mask = mask.expand_as(feat)
    cond_mask = cond_mask.expand_as(cond_feat)
    feat[mask] = adain_v(feat[mask].reshape(4,-1).transpose(0,1), cond_feat[cond_mask].reshape(4,-1).transpose(0,1), eps).transpose(0,1).flatten()
    return feat

def custom_mean(block, axis):
    return np.sum(block, axis) / np.maximum(np.sum(block != 0, axis), 1)


def preprocess(image):
    if isinstance(image, torch.Tensor):
        return image
    elif isinstance(image, PIL.Image.Image):
        image = [image]

    if isinstance(image[0], PIL.Image.Image):
        w, h = image[0].size
        w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32

        image = [
            np.array(i.resize((w, h), resample=PIL_INTERPOLATION["lanczos"]))[None, :]
            for i in image
        ]
        image = np.concatenate(image, axis=0)
        image = np.array(image).astype(np.float32) / 255.0
        image = image.transpose(0, 3, 1, 2)
        image = 2.0 * image - 1.0
        image = torch.from_numpy(image)
    elif isinstance(image[0], torch.Tensor):
        image = torch.cat(image, dim=0)
    return image


def preprocess_mask(mask, scale_factor=8, mask_blend_kernel=-1):
    w, h = mask.size
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    mask = mask.resize(
        (w // scale_factor, h // scale_factor), resample=PIL_INTERPOLATION["nearest"]
    )
    # if mask_blend_kernel > 0:
    #     mask = PIL.ImageOps.invert(
    #         blend_mask(PIL.ImageOps.invert(mask), kernel_size=mask_blend_kernel)
    #     )
    mask = np.array(mask).astype(np.float32) / 255.0
    mask = np.tile(mask, (4, 1, 1))
    mask = mask[None].transpose(0, 1, 2, 3)  # what does this step do?
    mask = 1 - mask  # repaint white, keep black
    mask = torch.from_numpy(mask)
    return mask

class ControlNetPipeline(StableDiffusionControlNetImg2ImgPipeline):

    def prepare_latents(
        self,
        image,
        timestep,
        batch_size,
        num_images_per_prompt,
        dtype,
        device,
        generator=None,
        latents=None,
        mask_img=None,
        latent_blend_kernel=-1,
    ):
        image = image.to(device=device, dtype=dtype)

        batch_size = batch_size * num_images_per_prompt
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if isinstance(generator, list):
            init_latents = [
                self.vae.encode(image[i : i + 1]).latent_dist.sample(generator[i])
                for i in range(batch_size)
            ]
            init_latents = torch.cat(init_latents, dim=0)
        else:
            init_latents = self.vae.encode(image).latent_dist.sample(generator)

        init_latents = 0.18215 * init_latents  # [1, 4, 64, 64]
        if mask_img is not None and latent_blend_kernel > 0:
            init_latents = init_latents.detach().cpu().numpy()
            init_latents_min = init_latents.min()
            init_latents_max = init_latents.max()
            init_latents = (init_latents - init_latents_min) / (
                init_latents_max - init_latents_min
            )
            init_latents_uint8 = (init_latents * 255).astype(np.uint8)
            for i in range(4):
                latents_out = blend_img(
                    init_latents_uint8[0, i],
                    (mask_img),
                    kernel_size=latent_blend_kernel,
                )
                init_latents_uint8[0, i] = latents_out

            init_latents = torch.tensor(init_latents_uint8).to(dtype).to(device) / 255
            init_latents = (
                init_latents * (init_latents_max - init_latents_min) + init_latents_min
            )

        if (
            batch_size > init_latents.shape[0]
            and batch_size % init_latents.shape[0] == 0
        ):
            # expand init_latents for batch_size
            deprecation_message = (
                f"You have passed {batch_size} text prompts (`prompt`), but only {init_latents.shape[0]} initial"
                " images (`image`). Initial images are now duplicating to match the number of text prompts. Note"
                " that this behavior is deprecated and will be removed in a version 1.0.0. Please make sure to update"
                " your script to pass as many initial images as text prompts to suppress this warning."
            )
            deprecate(
                "len(prompt) != len(image)",
                "1.0.0",
                deprecation_message,
                standard_warn=False,
            )
            additional_image_per_prompt = batch_size // init_latents.shape[0]
            init_latents = torch.cat(
                [init_latents] * additional_image_per_prompt, dim=0
            )
        elif (
            batch_size > init_latents.shape[0]
            and batch_size % init_latents.shape[0] != 0
        ):
            raise ValueError(
                f"Cannot duplicate `image` of batch size {init_latents.shape[0]} to {batch_size} text prompts."
            )
        else:
            init_latents = torch.cat([init_latents], dim=0)

        rand_device = "cpu" if device.type == "mps" else device
        shape = init_latents.shape
        if latents == None:
            if isinstance(generator, list):
                shape = (1,) + shape[1:]
                noise = [
                    torch.randn(
                        shape, generator=generator[i], device=rand_device, dtype=dtype
                    )
                    for i in range(batch_size)
                ]
                noise = torch.cat(noise, dim=0).to(device)
            else:
                noise = torch.randn(
                    shape, generator=generator, device=rand_device, dtype=dtype
                ).to(device)
        if latents != None:
            noise = latents
        # get latents
        latents_img_ori = init_latents
        init_latents = self.scheduler.add_noise(init_latents, noise, timestep)
        latents_img = init_latents
        return latents_img, latents_img_ori
    
    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        image: Union[torch.FloatTensor, PIL.Image.Image] = None,
        control_image: Union[torch.FloatTensor, PIL.Image.Image] = None,
        latent_code: Union[torch.FloatTensor, PIL.Image.Image] = None,
        latent_list: Union[torch.FloatTensor, PIL.Image.Image] = None,
        mask_image: Union[torch.FloatTensor, PIL.Image.Image] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        strength: float = 0.8,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_conditioning_scale: Union[float, List[float]] = 0.8,
        guess_mode: bool = False,
        control_guidance_start: Union[float, List[float]] = 0.0,
        control_guidance_end: Union[float, List[float]] = 1.0,
        clip_skip: Optional[int] = None,
        latent_blend_kernel: Optional[int] = -1,
        inpainting_strength: Optional[float] = 1,
        mask_blend_kernel: Optional[int] = -1,
        align : bool = False,
        obj_mask=None, obj_mask_src=None, mask_align=False,
        noise_align=True,
        start_align=True,
        latent_align=False,
        tg_latent_list=None,
        noise_list= None,
        desc: Optional[str] = None,
        align_var=False,
        repeat_loop : bool = False,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            image (`torch.FloatTensor`, `PIL.Image.Image`, `np.ndarray`, `List[torch.FloatTensor]`, `List[PIL.Image.Image]`, `List[np.ndarray]`,:
                    `List[List[torch.FloatTensor]]`, `List[List[np.ndarray]]` or `List[List[PIL.Image.Image]]`):
                The initial image to be used as the starting point for the image generation process. Can also accept
                image latents as `image`, and if passing latents directly they are not encoded again.
            control_image (`torch.FloatTensor`, `PIL.Image.Image`, `np.ndarray`, `List[torch.FloatTensor]`, `List[PIL.Image.Image]`, `List[np.ndarray]`,:
                    `List[List[torch.FloatTensor]]`, `List[List[np.ndarray]]` or `List[List[PIL.Image.Image]]`):
                The ControlNet input condition to provide guidance to the `unet` for generation. If the type is
                specified as `torch.FloatTensor`, it is passed to ControlNet as is. `PIL.Image.Image` can also be
                accepted as an image. The dimensions of the output image defaults to `image`'s dimensions. If height
                and/or width are passed, `image` is resized accordingly. If multiple ControlNets are specified in
                `init`, images must be passed as a list such that each element of the list can be correctly batched for
                input to a single ControlNet.
            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that calls every `callback_steps` steps during inference. The function is called with the
                following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function is called. If not specified, the callback is called at
                every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            controlnet_conditioning_scale (`float` or `List[float]`, *optional*, defaults to 1.0):
                The outputs of the ControlNet are multiplied by `controlnet_conditioning_scale` before they are added
                to the residual in the original `unet`. If multiple ControlNets are specified in `init`, you can set
                the corresponding scale as a list.
            guess_mode (`bool`, *optional*, defaults to `False`):
                The ControlNet encoder tries to recognize the content of the input image even if you remove all
                prompts. A `guidance_scale` value between 3.0 and 5.0 is recommended.
            control_guidance_start (`float` or `List[float]`, *optional*, defaults to 0.0):
                The percentage of total steps at which the ControlNet starts applying.
            control_guidance_end (`float` or `List[float]`, *optional*, defaults to 1.0):
                The percentage of total steps at which the ControlNet stops applying.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list with the generated images and the
                second element is a list of `bool`s indicating whether the corresponding generated image contains
                "not-safe-for-work" (nsfw) content.
        """
        controlnet = self.controlnet._orig_mod if is_compiled_module(self.controlnet) else self.controlnet

        # align format for control guidance
        if not isinstance(control_guidance_start, list) and isinstance(control_guidance_end, list):
            control_guidance_start = len(control_guidance_end) * [control_guidance_start]
        elif not isinstance(control_guidance_end, list) and isinstance(control_guidance_start, list):
            control_guidance_end = len(control_guidance_start) * [control_guidance_end]
        elif not isinstance(control_guidance_start, list) and not isinstance(control_guidance_end, list):
            mult = len(controlnet.nets) if isinstance(controlnet, MultiControlNetModel) else 1
            control_guidance_start, control_guidance_end = mult * [control_guidance_start], mult * [
                control_guidance_end
            ]

        # 1. Check inputs. Raise error if not correct
        # self.check_inputs(
        #     prompt,
        #     control_image,
        #     callback_steps,
        #     negative_prompt,
        #     prompt_embeds,
        #     negative_prompt_embeds,
        #     controlnet_conditioning_scale,
        #     control_guidance_start,
        #     control_guidance_end,
        # )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        if isinstance(controlnet, MultiControlNetModel) and isinstance(controlnet_conditioning_scale, float):
            controlnet_conditioning_scale = [controlnet_conditioning_scale] * len(controlnet.nets)

        global_pool_conditions = (
            controlnet.config.global_pool_conditions
            if isinstance(controlnet, ControlNetModel)
            else controlnet.nets[0].config.global_pool_conditions
        )
        guess_mode = guess_mode or global_pool_conditions

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
            clip_skip=clip_skip,
        )
        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        # 4. Prepare image
        image = self.image_processor.preprocess(image, height=height, width=width).to(dtype=torch.float32)

        # 5. Prepare controlnet_conditioning_image
        if isinstance(controlnet, ControlNetModel):
            control_image = self.prepare_control_image(
                image=control_image,
                width=width,
                height=height,
                batch_size=batch_size * num_images_per_prompt,
                num_images_per_prompt=num_images_per_prompt,
                device=device,
                dtype=controlnet.dtype,
                do_classifier_free_guidance=do_classifier_free_guidance,
                guess_mode=guess_mode,
            )
        elif isinstance(controlnet, MultiControlNetModel):
            control_images = []

            for control_image_ in control_image:
                control_image_ = self.prepare_control_image(
                    image=control_image_,
                    width=width,
                    height=height,
                    batch_size=batch_size * num_images_per_prompt,
                    num_images_per_prompt=num_images_per_prompt,
                    device=device,
                    dtype=controlnet.dtype,
                    do_classifier_free_guidance=do_classifier_free_guidance,
                    guess_mode=guess_mode,
                )
                # if len(latent_code) == 2:
                #     control_image_ = torch.cat([control_image_])
                control_images.append(control_image_)

            control_image = control_images
        else:
            assert False

        # 5. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength, device)
        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)

        # 6. Prepare latent variables
        latents_img, latents_img_ori = self.prepare_latents(
                image,
                latent_timestep,
                batch_size,
                num_images_per_prompt,
                prompt_embeds.dtype,
                device,
                generator,
                latents,
                mask_img=None,
                latent_blend_kernel=latent_blend_kernel,
            )
        
        if latent_code is not None:
            if isinstance(latent_code, list):
                latents_img = torch.cat(latent_code, dim=0)
            else :
                latents_img = latent_code

        # 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7.1 Create tensor stating which controlnets to keep
        controlnet_keep = []
        for i in range(len(timesteps)):
            keeps = [
                1.0 - float(i / len(timesteps) < s or (i + 1) / len(timesteps) > e)
                for s, e in zip(control_guidance_start, control_guidance_end)
            ]
            controlnet_keep.append(keeps[0] if isinstance(controlnet, ControlNetModel) else keeps)
        
        if mask_image is not None:
            if not isinstance(mask_image, torch.FloatTensor):
                # mask_image_clone = mask_image.copy()
                # vae_mask = preprocess_mask(
                #     mask_image_clone, 1, mask_blend_kernel=mask_blend_kernel
                # )
                print('nonononononon')
                mask_image = preprocess_mask(
                    mask_image, self.vae_scale_factor, mask_blend_kernel=mask_blend_kernel
                )
            mask = mask_image.to(device=self.device, dtype=latents_img.dtype)
            mask_cat_ = torch.cat([mask] * batch_size * num_images_per_prompt)

        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        t0 = timesteps[0]
        #with self.progress_bar(total=num_inference_steps) as progress_bar:
        with tqdm(total=num_inference_steps, desc=desc) as progress_bar:
            for i, t in enumerate(timesteps):
                # if repeat_loop:
                #     if  i>=15 and i<=35:
                #         repeat = 3
                #     else :
                #         repeat = 1
                # else :
                #     repeat = 1
                #for j in range(repeat): 
                # mask_cat = (1-t0/1000) + mask_cat*(t0/1000)>  1 - t/1000
                mask_cat = (t > t0*(1-mask_cat_)).to(dtype=latents_img.dtype)
                # mask_cat = (mask_cat_ > 1 - t/1000).to(dtype=latents_img.dtype) if mask_image is not None else None
                
                if latent_list is not None:
                    latents_img[:1] = latent_list[0][-1-i]
                    
                if align and start_align and i==0:
                    B,C,H,W = latents_img.shape
                    print(latents_img.shape)
                    if mask_align:
                        latents_img[1:] = masked_adain(latents_img[1:], latents_img[:1], obj_mask, obj_mask_src)
                    else:
                        latents_img[1:] = adain(latents_img[1:].permute(0,2,3,1).reshape(-1, C), latents_img[:1].permute(0,2,3,1).reshape(-1, C)).reshape(1,H,W,C).permute(0,3,1,2).contiguous()
                    
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents_img] * 2) if do_classifier_free_guidance else latents_img
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # controlnet(s) inference
                if guess_mode and do_classifier_free_guidance:
                    # Infer ControlNet only for the conditional batch.
                    control_model_input = latents_img
                    control_model_input = self.scheduler.scale_model_input(control_model_input, t)
                    controlnet_prompt_embeds = prompt_embeds.chunk(2)[1]
                else:
                    control_model_input = latent_model_input
                    controlnet_prompt_embeds = prompt_embeds

                if isinstance(controlnet_keep[i], list):
                    cond_scale = [c * s for c, s in zip(controlnet_conditioning_scale, controlnet_keep[i])]
                else:
                    controlnet_cond_scale = controlnet_conditioning_scale
                    if isinstance(controlnet_cond_scale, list):
                        controlnet_cond_scale = controlnet_cond_scale[0]
                    cond_scale = controlnet_cond_scale * controlnet_keep[i]

                down_block_res_samples, mid_block_res_sample = self.controlnet(
                    control_model_input,
                    t,
                    encoder_hidden_states=controlnet_prompt_embeds,
                    controlnet_cond=control_image,
                    conditioning_scale=cond_scale,
                    guess_mode=guess_mode,
                    return_dict=False,
                )

                if guess_mode and do_classifier_free_guidance:
                    # Infered ControlNet only for the conditional batch.
                    # To apply the output of ControlNet to both the unconditional and conditional batches,
                    # add 0 to the unconditional batch to keep it unchanged.
                    down_block_res_samples = [torch.cat([torch.zeros_like(d), d]) for d in down_block_res_samples]
                    mid_block_res_sample = torch.cat([torch.zeros_like(mid_block_res_sample), mid_block_res_sample])

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                    return_dict=False,
                )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                    if align and noise_align and t > 1000*0.6:
                        B,C,H,W = noise_pred.shape
                        if mask_align:
                            noise_pred[1:] = masked_adain(noise_pred[1:], noise_list[-1-i], obj_mask, obj_mask_src)
                        else:
                            noise_pred[1:] = adain(noise_pred[1:].permute(0,2,3,1).reshape(-1, C), noise_list[-1-i].permute(0,2,3,1).reshape(-1, C)).reshape(1,H,W,C).permute(0,3,1,2).contiguous()
                        # noise_pred[1:] = 

                # compute the previous noisy sample x_t -> x_t-1
                #latents_img = self.scheduler.step(noise_pred, t, latents_img, **extra_step_kwargs, return_dict=False)[0]
                latents_img = self.step(noise_pred, t, latents_img)
                    # if j < repeat -1:
                    #     latents_img = self.next_step(noise_pred, t, latents_img)

                if latent_list is not None:
                    
                    if align and latent_align and t > 1000*0.6:
                        B,C,H,W = noise_pred.shape
                        if mask_align:
                            latents_img[1:] = masked_adain(latents_img[1:], latents_img[:1], obj_mask, obj_mask_src)
                        else:
                            latents_img[1:] = adain(latents_img[1:].permute(0,2,3,1).reshape(-1, C), latents_img[:1].permute(0,2,3,1).reshape(-1, C)).reshape(1,H,W,C).permute(0,3,1,2).contiguous()
                    
                    #latents_img[:1, :, :, :] = latent_list[0][-i-2]
                    if t > 1000 * (1 - inpainting_strength) and mask_image is not None:
                        latents_img[1:2, :, :, :]= (latent_list[1][-i-2] * mask_cat[1:2]) + (
                            latents_img[1:2, :, :, :] * (1 - mask_cat[1:2])
                        )
                        if align_var:
                            latents_img[1:2, :, :, :] = masked_adain_v(latents_img[1:2, :, :, :], latent_list[1][-i-2], (1 - mask_cat[1:2]).bool(), (1 - mask_cat[1:2]).bool())
                        
                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

        # If we do sequential model offloading, let's offload unet and controlnet
        # manually for max memory savings
        # if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
        #     self.unet.to("cpu")
        #     self.controlnet.to("cpu")
        #     torch.cuda.empty_cache()

        if not output_type == "latent":
            image = self.vae.decode(latents_img / self.vae.config.scaling_factor, return_dict=False)[0]
            #image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
        else:
            image = latents
            #has_nsfw_concept = None

        do_denormalize = [True] * image.shape[0]
    
        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        # Offload all models
        # self.maybe_free_model_hooks()

        # if not return_dict:
        #     return (image, has_nsfw_concept)

        return ImagePipelineOutput(images=image)
    
    @torch.no_grad()
    def vaecall(
        self,
        prompt: Union[str, List[str]] = None,
        image: Union[torch.FloatTensor, PIL.Image.Image] = None,
        control_image: Union[torch.FloatTensor, PIL.Image.Image] = None,
        latent_code: Union[torch.FloatTensor, PIL.Image.Image] = None,
        latent_list: Union[torch.FloatTensor, PIL.Image.Image] = None,
        mask_image: Union[torch.FloatTensor, PIL.Image.Image] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        strength: float = 0.8,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_conditioning_scale: Union[float, List[float]] = 0.8,
        guess_mode: bool = False,
        control_guidance_start: Union[float, List[float]] = 0.0,
        control_guidance_end: Union[float, List[float]] = 1.0,
        clip_skip: Optional[int] = None,
        latent_blend_kernel: Optional[int] = -1,
        inpainting_strength: Optional[float] = 1,
        mask_blend_kernel: Optional[int] = -1,
        align : bool = False,
        obj_mask=None, obj_mask_src=None, mask_align=False,
        noise_align=True,
        start_align=True,
        latent_align=False,
        tg_latent_list=None,
        noise_list= None,
        desc: Optional[str] = None,
        align_var=False,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            image (`torch.FloatTensor`, `PIL.Image.Image`, `np.ndarray`, `List[torch.FloatTensor]`, `List[PIL.Image.Image]`, `List[np.ndarray]`,:
                    `List[List[torch.FloatTensor]]`, `List[List[np.ndarray]]` or `List[List[PIL.Image.Image]]`):
                The initial image to be used as the starting point for the image generation process. Can also accept
                image latents as `image`, and if passing latents directly they are not encoded again.
            control_image (`torch.FloatTensor`, `PIL.Image.Image`, `np.ndarray`, `List[torch.FloatTensor]`, `List[PIL.Image.Image]`, `List[np.ndarray]`,:
                    `List[List[torch.FloatTensor]]`, `List[List[np.ndarray]]` or `List[List[PIL.Image.Image]]`):
                The ControlNet input condition to provide guidance to the `unet` for generation. If the type is
                specified as `torch.FloatTensor`, it is passed to ControlNet as is. `PIL.Image.Image` can also be
                accepted as an image. The dimensions of the output image defaults to `image`'s dimensions. If height
                and/or width are passed, `image` is resized accordingly. If multiple ControlNets are specified in
                `init`, images must be passed as a list such that each element of the list can be correctly batched for
                input to a single ControlNet.
            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that calls every `callback_steps` steps during inference. The function is called with the
                following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function is called. If not specified, the callback is called at
                every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            controlnet_conditioning_scale (`float` or `List[float]`, *optional*, defaults to 1.0):
                The outputs of the ControlNet are multiplied by `controlnet_conditioning_scale` before they are added
                to the residual in the original `unet`. If multiple ControlNets are specified in `init`, you can set
                the corresponding scale as a list.
            guess_mode (`bool`, *optional*, defaults to `False`):
                The ControlNet encoder tries to recognize the content of the input image even if you remove all
                prompts. A `guidance_scale` value between 3.0 and 5.0 is recommended.
            control_guidance_start (`float` or `List[float]`, *optional*, defaults to 0.0):
                The percentage of total steps at which the ControlNet starts applying.
            control_guidance_end (`float` or `List[float]`, *optional*, defaults to 1.0):
                The percentage of total steps at which the ControlNet stops applying.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list with the generated images and the
                second element is a list of `bool`s indicating whether the corresponding generated image contains
                "not-safe-for-work" (nsfw) content.
        """
        controlnet = self.controlnet._orig_mod if is_compiled_module(self.controlnet) else self.controlnet

        # align format for control guidance
        if not isinstance(control_guidance_start, list) and isinstance(control_guidance_end, list):
            control_guidance_start = len(control_guidance_end) * [control_guidance_start]
        elif not isinstance(control_guidance_end, list) and isinstance(control_guidance_start, list):
            control_guidance_end = len(control_guidance_start) * [control_guidance_end]
        elif not isinstance(control_guidance_start, list) and not isinstance(control_guidance_end, list):
            mult = len(controlnet.nets) if isinstance(controlnet, MultiControlNetModel) else 1
            control_guidance_start, control_guidance_end = mult * [control_guidance_start], mult * [
                control_guidance_end
            ]

        # 1. Check inputs. Raise error if not correct
        # self.check_inputs(
        #     prompt,
        #     control_image,
        #     callback_steps,
        #     negative_prompt,
        #     prompt_embeds,
        #     negative_prompt_embeds,
        #     controlnet_conditioning_scale,
        #     control_guidance_start,
        #     control_guidance_end,
        # )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        if isinstance(controlnet, MultiControlNetModel) and isinstance(controlnet_conditioning_scale, float):
            controlnet_conditioning_scale = [controlnet_conditioning_scale] * len(controlnet.nets)

        global_pool_conditions = (
            controlnet.config.global_pool_conditions
            if isinstance(controlnet, ControlNetModel)
            else controlnet.nets[0].config.global_pool_conditions
        )
        guess_mode = guess_mode or global_pool_conditions

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
            clip_skip=clip_skip,
        )
        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        # 4. Prepare image
        image = self.image_processor.preprocess(image, height=height, width=width).to(dtype=torch.float32)

        # 5. Prepare controlnet_conditioning_image
        if isinstance(controlnet, ControlNetModel):
            control_image = self.prepare_control_image(
                image=control_image,
                width=width,
                height=height,
                batch_size=batch_size * num_images_per_prompt,
                num_images_per_prompt=num_images_per_prompt,
                device=device,
                dtype=controlnet.dtype,
                do_classifier_free_guidance=do_classifier_free_guidance,
                guess_mode=guess_mode,
            )
        elif isinstance(controlnet, MultiControlNetModel):
            control_images = []

            for control_image_ in control_image:
                control_image_ = self.prepare_control_image(
                    image=control_image_,
                    width=width,
                    height=height,
                    batch_size=batch_size * num_images_per_prompt,
                    num_images_per_prompt=num_images_per_prompt,
                    device=device,
                    dtype=controlnet.dtype,
                    do_classifier_free_guidance=do_classifier_free_guidance,
                    guess_mode=guess_mode,
                )
                # if len(latent_code) == 2:
                #     control_image_ = torch.cat([control_image_])
                control_images.append(control_image_)

            control_image = control_images
        else:
            assert False

        # 5. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength, device)
        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)

        # 6. Prepare latent variables
        latents_img, latents_img_ori = self.prepare_latents(
                image,
                latent_timestep,
                batch_size,
                num_images_per_prompt,
                prompt_embeds.dtype,
                device,
                generator,
                latents,
                mask_img=None,
                latent_blend_kernel=latent_blend_kernel,
            )
        
        if latent_code is not None:
            if isinstance(latent_code, list):
                latents_img = torch.cat(latent_code, dim=0)
            else :
                latents_img = latent_code

        # 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7.1 Create tensor stating which controlnets to keep
        controlnet_keep = []
        for i in range(len(timesteps)):
            keeps = [
                1.0 - float(i / len(timesteps) < s or (i + 1) / len(timesteps) > e)
                for s, e in zip(control_guidance_start, control_guidance_end)
            ]
            controlnet_keep.append(keeps[0] if isinstance(controlnet, ControlNetModel) else keeps)
        
        if mask_image is not None:
            if not isinstance(mask_image, torch.FloatTensor):
                mask_image_clone = mask_image.copy()
                vae_mask = preprocess_mask(
                    mask_image_clone, 1, mask_blend_kernel=mask_blend_kernel
                )
                mask_image = preprocess_mask(
                    mask_image, self.vae_scale_factor, mask_blend_kernel=mask_blend_kernel
                )
            vae_mask = vae_mask.to(device=self.device, dtype=latents_img.dtype)
            mask = mask_image.to(device=self.device, dtype=latents_img.dtype)
            mask_cat = torch.cat([mask] * batch_size * num_images_per_prompt)

        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        #with self.progress_bar(total=num_inference_steps) as progress_bar:
        with tqdm(total=num_inference_steps, desc=desc) as progress_bar:
            for i, t in enumerate(timesteps):
                if latent_list is not None:
                    latents_img[:1] = latent_list[0][-1-i]
                    
                if align and start_align and i==0:
                    B,C,H,W = latents_img.shape
                    print(latents_img.shape)
                    if mask_align:
                        latents_img[1:] = masked_adain(latents_img[1:], latents_img[:1], obj_mask, obj_mask_src)
                    else:
                        latents_img[1:] = adain(latents_img[1:].permute(0,2,3,1).reshape(-1, C), latents_img[:1].permute(0,2,3,1).reshape(-1, C)).reshape(1,H,W,C).permute(0,3,1,2).contiguous()
                    
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents_img] * 2) if do_classifier_free_guidance else latents_img
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # controlnet(s) inference
                if guess_mode and do_classifier_free_guidance:
                    # Infer ControlNet only for the conditional batch.
                    control_model_input = latents_img
                    control_model_input = self.scheduler.scale_model_input(control_model_input, t)
                    controlnet_prompt_embeds = prompt_embeds.chunk(2)[1]
                else:
                    control_model_input = latent_model_input
                    controlnet_prompt_embeds = prompt_embeds

                if isinstance(controlnet_keep[i], list):
                    cond_scale = [c * s for c, s in zip(controlnet_conditioning_scale, controlnet_keep[i])]
                else:
                    controlnet_cond_scale = controlnet_conditioning_scale
                    if isinstance(controlnet_cond_scale, list):
                        controlnet_cond_scale = controlnet_cond_scale[0]
                    cond_scale = controlnet_cond_scale * controlnet_keep[i]

                down_block_res_samples, mid_block_res_sample = self.controlnet(
                    control_model_input,
                    t,
                    encoder_hidden_states=controlnet_prompt_embeds,
                    controlnet_cond=control_image,
                    conditioning_scale=cond_scale,
                    guess_mode=guess_mode,
                    return_dict=False,
                )

                if guess_mode and do_classifier_free_guidance:
                    # Infered ControlNet only for the conditional batch.
                    # To apply the output of ControlNet to both the unconditional and conditional batches,
                    # add 0 to the unconditional batch to keep it unchanged.
                    down_block_res_samples = [torch.cat([torch.zeros_like(d), d]) for d in down_block_res_samples]
                    mid_block_res_sample = torch.cat([torch.zeros_like(mid_block_res_sample), mid_block_res_sample])

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                    return_dict=False,
                )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                    if align and noise_align and t > 1000*0.6:
                        B,C,H,W = noise_pred.shape
                        if mask_align:
                            noise_pred[1:] = masked_adain(noise_pred[1:], noise_list[-1-i], obj_mask, obj_mask_src)
                        else:
                            noise_pred[1:] = adain(noise_pred[1:].permute(0,2,3,1).reshape(-1, C), noise_list[-1-i].permute(0,2,3,1).reshape(-1, C)).reshape(1,H,W,C).permute(0,3,1,2).contiguous()
                        # noise_pred[1:] = 

                # compute the previous noisy sample x_t -> x_t-1
                latents_img = self.scheduler.step(noise_pred, t, latents_img, **extra_step_kwargs, return_dict=False)[0]
                
                if latent_list is not None:
                    
                    if align and latent_align and t > 1000*0.6:
                        B,C,H,W = noise_pred.shape
                        if mask_align:
                            latents_img[1:] = masked_adain(latents_img[1:], latents_img[:1], obj_mask, obj_mask_src)
                        else:
                            latents_img[1:] = adain(latents_img[1:].permute(0,2,3,1).reshape(-1, C), latents_img[:1].permute(0,2,3,1).reshape(-1, C)).reshape(1,H,W,C).permute(0,3,1,2).contiguous()
                    
                    #latents_img[:1, :, :, :] = latent_list[0][-i-2]
                    if t > 1000 * (1 - inpainting_strength) and mask_image is not None:
                        latents_img[1:2, :, :, :]= (latent_list[1][-i-2] * mask_cat[1:2]) + (
                            latents_img[1:2, :, :, :] * (1 - mask_cat[1:2])
                        )
                        if align_var:
                            latents_img[1:2, :, :, :] = masked_adain_v(latents_img[1:2, :, :, :], latent_list[1][-i-2], (1 - mask_cat[1:2]).bool(), (1 - mask_cat[1:2]).bool())
                        
                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

        # If we do sequential model offloading, let's offload unet and controlnet
        # manually for max memory savings
        # if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
        #     self.unet.to("cpu")
        #     self.controlnet.to("cpu")
        #     torch.cuda.empty_cache()

        if not output_type == "latent":
            if latents_img.shape[0] ==2 :
                latents_img = latents_img[1:]
            vae_image, vae_list= self.vae.decode(latent_list[1][0] / self.vae.config.scaling_factor, return_dict=False, return_list=True)
            image = self.vae.decode(latents_img / self.vae.config.scaling_factor, return_dict=False, mask=vae_mask, vae_list=vae_list)[0]
            #image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
        else:
            image = latents
            #has_nsfw_concept = None

        do_denormalize = [True] * image.shape[0]
    
        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        # Offload all models
        # self.maybe_free_model_hooks()

        # if not return_dict:
        #     return (image, has_nsfw_concept)

        return ImagePipelineOutput(images=image)
    
    @torch.no_grad()
    def uncon_call_(
        self,
        prompt: Union[str, List[str]] = None,
        image: Union[torch.FloatTensor, PIL.Image.Image] = None,
        control_image: Union[torch.FloatTensor, PIL.Image.Image] = None,
        latent_code: Union[torch.FloatTensor, PIL.Image.Image] = None,
        latent_list: Union[torch.FloatTensor, PIL.Image.Image] = None,
        mask_image: Union[torch.FloatTensor, PIL.Image.Image] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        strength: float = 0.8,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_conditioning_scale: Union[float, List[float]] = 0.8,
        guess_mode: bool = False,
        control_guidance_start: Union[float, List[float]] = 0.0,
        control_guidance_end: Union[float, List[float]] = 1.0,
        clip_skip: Optional[int] = None,
        latent_blend_kernel: Optional[int] = -1,
        inpainting_strength: Optional[float] = 1,
        mask_blend_kernel: Optional[int] = -1,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            image (`torch.FloatTensor`, `PIL.Image.Image`, `np.ndarray`, `List[torch.FloatTensor]`, `List[PIL.Image.Image]`, `List[np.ndarray]`,:
                    `List[List[torch.FloatTensor]]`, `List[List[np.ndarray]]` or `List[List[PIL.Image.Image]]`):
                The initial image to be used as the starting point for the image generation process. Can also accept
                image latents as `image`, and if passing latents directly they are not encoded again.
            control_image (`torch.FloatTensor`, `PIL.Image.Image`, `np.ndarray`, `List[torch.FloatTensor]`, `List[PIL.Image.Image]`, `List[np.ndarray]`,:
                    `List[List[torch.FloatTensor]]`, `List[List[np.ndarray]]` or `List[List[PIL.Image.Image]]`):
                The ControlNet input condition to provide guidance to the `unet` for generation. If the type is
                specified as `torch.FloatTensor`, it is passed to ControlNet as is. `PIL.Image.Image` can also be
                accepted as an image. The dimensions of the output image defaults to `image`'s dimensions. If height
                and/or width are passed, `image` is resized accordingly. If multiple ControlNets are specified in
                `init`, images must be passed as a list such that each element of the list can be correctly batched for
                input to a single ControlNet.
            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that calls every `callback_steps` steps during inference. The function is called with the
                following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function is called. If not specified, the callback is called at
                every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            controlnet_conditioning_scale (`float` or `List[float]`, *optional*, defaults to 1.0):
                The outputs of the ControlNet are multiplied by `controlnet_conditioning_scale` before they are added
                to the residual in the original `unet`. If multiple ControlNets are specified in `init`, you can set
                the corresponding scale as a list.
            guess_mode (`bool`, *optional*, defaults to `False`):
                The ControlNet encoder tries to recognize the content of the input image even if you remove all
                prompts. A `guidance_scale` value between 3.0 and 5.0 is recommended.
            control_guidance_start (`float` or `List[float]`, *optional*, defaults to 0.0):
                The percentage of total steps at which the ControlNet starts applying.
            control_guidance_end (`float` or `List[float]`, *optional*, defaults to 1.0):
                The percentage of total steps at which the ControlNet stops applying.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list with the generated images and the
                second element is a list of `bool`s indicating whether the corresponding generated image contains
                "not-safe-for-work" (nsfw) content.
        """
        #controlnet = self.controlnet._orig_mod if is_compiled_module(self.controlnet) else self.controlnet

        # align format for control guidance
        # if not isinstance(control_guidance_start, list) and isinstance(control_guidance_end, list):
        #     control_guidance_start = len(control_guidance_end) * [control_guidance_start]
        # elif not isinstance(control_guidance_end, list) and isinstance(control_guidance_start, list):
        #     control_guidance_end = len(control_guidance_start) * [control_guidance_end]
        # elif not isinstance(control_guidance_start, list) and not isinstance(control_guidance_end, list):
        #     mult = len(controlnet.nets) if isinstance(controlnet, MultiControlNetModel) else 1
        #     control_guidance_start, control_guidance_end = mult * [control_guidance_start], mult * [
        #         control_guidance_end
        #     ]

        # 1. Check inputs. Raise error if not correct
        # self.check_inputs(
        #     prompt,
        #     control_image,
        #     callback_steps,
        #     negative_prompt,
        #     prompt_embeds,
        #     negative_prompt_embeds,
        #     controlnet_conditioning_scale,
        #     control_guidance_start,
        #     control_guidance_end,
        # )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # if isinstance(controlnet, MultiControlNetModel) and isinstance(controlnet_conditioning_scale, float):
        #     controlnet_conditioning_scale = [controlnet_conditioning_scale] * len(controlnet.nets)

        # global_pool_conditions = (
        #     controlnet.config.global_pool_conditions
        #     if isinstance(controlnet, ControlNetModel)
        #     else controlnet.nets[0].config.global_pool_conditions
        # )
        # guess_mode = guess_mode or global_pool_conditions

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
            clip_skip=clip_skip,
        )
        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        # 4. Prepare image
        image = self.image_processor.preprocess(image, height=height, width=width).to(dtype=torch.float32)

        # 5. Prepare controlnet_conditioning_image
        # if isinstance(controlnet, ControlNetModel):
        #     control_image = self.prepare_control_image(
        #         image=control_image,
        #         width=width,
        #         height=height,
        #         batch_size=batch_size * num_images_per_prompt,
        #         num_images_per_prompt=num_images_per_prompt,
        #         device=device,
        #         dtype=controlnet.dtype,
        #         do_classifier_free_guidance=do_classifier_free_guidance,
        #         guess_mode=guess_mode,
        #     )
        # elif isinstance(controlnet, MultiControlNetModel):
        #     control_images = []

        #     for control_image_ in control_image:
        #         control_image_ = self.prepare_control_image(
        #             image=control_image_,
        #             width=width,
        #             height=height,
        #             batch_size=batch_size * num_images_per_prompt,
        #             num_images_per_prompt=num_images_per_prompt,
        #             device=device,
        #             dtype=controlnet.dtype,
        #             do_classifier_free_guidance=do_classifier_free_guidance,
        #             guess_mode=guess_mode,
        #         )

        #         control_images.append(control_image_)

        #     control_image = control_images
        # else:
        #     assert False

        # 5. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength, device)
        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)

        # 6. Prepare latent variables
        latents_img, latents_img_ori = self.prepare_latents(
                image,
                latent_timestep,
                batch_size,
                num_images_per_prompt,
                prompt_embeds.dtype,
                device,
                generator,
                latents,
                mask_img=None,
                latent_blend_kernel=latent_blend_kernel,
            )
        
        if latent_code is not None:
            if isinstance(latent_code, list):
                latents_img = torch.cat(latent_code, dim=0)
            else :
                latents_img = latent_code
                
        # 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7.1 Create tensor stating which controlnets to keep
        # controlnet_keep = []
        # for i in range(len(timesteps)):
        #     keeps = [
        #         1.0 - float(i / len(timesteps) < s or (i + 1) / len(timesteps) > e)
        #         for s, e in zip(control_guidance_start, control_guidance_end)
        #     ]
        #     controlnet_keep.append(keeps[0] if isinstance(controlnet, ControlNetModel) else keeps)
        if mask_image is not None:
            if not isinstance(mask_image, torch.FloatTensor):
                # mask_image_clone = mask_image.copy()
                # vae_mask = preprocess_mask(
                #     mask_image_clone, 1, mask_blend_kernel=mask_blend_kernel
                # )
                mask_image = preprocess_mask(
                    mask_image, self.vae_scale_factor, mask_blend_kernel=mask_blend_kernel
                )
            mask = mask_image.to(device=self.device, dtype=latents_img.dtype)
            mask_cat = torch.cat([mask] * batch_size * num_images_per_prompt)
        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # if latent_list is not None:
                #     latents_img[:1] = latent_list[-1-i]
                    
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents_img] * 2) if do_classifier_free_guidance else latents_img
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # # controlnet(s) inference
                # if guess_mode and do_classifier_free_guidance:
                #     # Infer ControlNet only for the conditional batch.
                #     control_model_input = latents_img
                #     control_model_input = self.scheduler.scale_model_input(control_model_input, t)
                #     controlnet_prompt_embeds = prompt_embeds.chunk(2)[1]
                # else:
                #     control_model_input = latent_model_input
                #     controlnet_prompt_embeds = prompt_embeds

                # if isinstance(controlnet_keep[i], list):
                #     cond_scale = [c * s for c, s in zip(controlnet_conditioning_scale, controlnet_keep[i])]
                # else:
                #     controlnet_cond_scale = controlnet_conditioning_scale
                #     if isinstance(controlnet_cond_scale, list):
                #         controlnet_cond_scale = controlnet_cond_scale[0]
                #     cond_scale = controlnet_cond_scale * controlnet_keep[i]

                # down_block_res_samples, mid_block_res_sample = self.controlnet(
                #     control_model_input,
                #     t,
                #     encoder_hidden_states=controlnet_prompt_embeds,
                #     controlnet_cond=control_image,
                #     conditioning_scale=cond_scale,
                #     guess_mode=guess_mode,
                #     return_dict=False,
                # )

                # if guess_mode and do_classifier_free_guidance:
                #     # Infered ControlNet only for the conditional batch.
                #     # To apply the output of ControlNet to both the unconditional and conditional batches,
                #     # add 0 to the unconditional batch to keep it unchanged.
                #     down_block_res_samples = [torch.cat([torch.zeros_like(d), d]) for d in down_block_res_samples]
                #     mid_block_res_sample = torch.cat([torch.zeros_like(mid_block_res_sample), mid_block_res_sample])

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                    # down_block_additional_residuals=down_block_res_samples,
                    # mid_block_additional_residual=mid_block_res_sample,
                    return_dict=False,
                )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents_img = self.scheduler.step(noise_pred, t, latents_img, **extra_step_kwargs, return_dict=False)[0]

                if t > 1000 * (1 - inpainting_strength) and mask_image is not None:
                        latents_img[:1, :, :, :]= (latent_list[-i-2] * mask_cat) + (
                            latents_img[:1, :, :, :] * (1 - mask_cat)
                        )
                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

        # If we do sequential model offloading, let's offload unet and controlnet
        # manually for max memory savings
        # if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
        #     self.unet.to("cpu")
        #     self.controlnet.to("cpu")
        #     torch.cuda.empty_cache()

        if not output_type == "latent":
            image = self.vae.decode(latents_img / self.vae.config.scaling_factor, return_dict=False)[0]
            #image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
        else:
            image = latents_img
            #has_nsfw_concept = None

        do_denormalize = [True] * image.shape[0]
    
        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        # Offload all models
        # self.maybe_free_model_hooks()

        # if not return_dict:
        #     return (image, has_nsfw_concept)

        return ImagePipelineOutput(images=image)
    
    def next_step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        x: torch.FloatTensor,
        eta=0.,
        verbose=False
    ):
        """
        Inverse sampling for DDIM Inversion
        """
        if verbose:
            print("timestep: ", timestep)
        next_step = timestep
        timestep = min(timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999)
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_step]
        beta_prod_t = 1 - alpha_prod_t
        pred_x0 = (x - beta_prod_t**0.5 * model_output) / alpha_prod_t**0.5
        pred_dir = (1 - alpha_prod_t_next)**0.5 * model_output
        x_next = alpha_prod_t_next**0.5 * pred_x0 + pred_dir
        return x_next
    
    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        x: torch.FloatTensor,
        eta: float=0.0,
        verbose=False,
    ):
        """
        predict the sampe the next step in the denoise process.
        """
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep > 0 else self.scheduler.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        pred_x0 = (x - beta_prod_t**0.5 * model_output) / alpha_prod_t**0.5
        pred_dir = (1 - alpha_prod_t_prev)**0.5 * model_output
        x_prev = alpha_prod_t_prev**0.5 * pred_x0 + pred_dir
        return x_prev
    
    @torch.no_grad()
    def invert(
        self,
        prompt: Union[str, List[str]] = None,
        image: Union[torch.FloatTensor, PIL.Image.Image] = None,
        control_image: Union[torch.FloatTensor, PIL.Image.Image] = None,
        mask_image: Union[torch.FloatTensor, PIL.Image.Image] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        strength: float = 0.8,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_conditioning_scale: Union[float, List[float]] = 0.8,
        guess_mode: bool = False,
        control_guidance_start: Union[float, List[float]] = 0.0,
        control_guidance_end: Union[float, List[float]] = 1.0,
        clip_skip: Optional[int] = None,
        latent_blend_kernel: Optional[int] = -1,
        return_intermediates : bool = False,
        inpainting_strength: Optional[float] = 1,
        mask_blend_kernel: Optional[int] = -1,
        desc: Optional[str] = None,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            image (`torch.FloatTensor`, `PIL.Image.Image`, `np.ndarray`, `List[torch.FloatTensor]`, `List[PIL.Image.Image]`, `List[np.ndarray]`,:
                    `List[List[torch.FloatTensor]]`, `List[List[np.ndarray]]` or `List[List[PIL.Image.Image]]`):
                The initial image to be used as the starting point for the image generation process. Can also accept
                image latents as `image`, and if passing latents directly they are not encoded again.
            control_image (`torch.FloatTensor`, `PIL.Image.Image`, `np.ndarray`, `List[torch.FloatTensor]`, `List[PIL.Image.Image]`, `List[np.ndarray]`,:
                    `List[List[torch.FloatTensor]]`, `List[List[np.ndarray]]` or `List[List[PIL.Image.Image]]`):
                The ControlNet input condition to provide guidance to the `unet` for generation. If the type is
                specified as `torch.FloatTensor`, it is passed to ControlNet as is. `PIL.Image.Image` can also be
                accepted as an image. The dimensions of the output image defaults to `image`'s dimensions. If height
                and/or width are passed, `image` is resized accordingly. If multiple ControlNets are specified in
                `init`, images must be passed as a list such that each element of the list can be correctly batched for
                input to a single ControlNet.
            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that calls every `callback_steps` steps during inference. The function is called with the
                following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function is called. If not specified, the callback is called at
                every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            controlnet_conditioning_scale (`float` or `List[float]`, *optional*, defaults to 1.0):
                The outputs of the ControlNet are multiplied by `controlnet_conditioning_scale` before they are added
                to the residual in the original `unet`. If multiple ControlNets are specified in `init`, you can set
                the corresponding scale as a list.
            guess_mode (`bool`, *optional*, defaults to `False`):
                The ControlNet encoder tries to recognize the content of the input image even if you remove all
                prompts. A `guidance_scale` value between 3.0 and 5.0 is recommended.
            control_guidance_start (`float` or `List[float]`, *optional*, defaults to 0.0):
                The percentage of total steps at which the ControlNet starts applying.
            control_guidance_end (`float` or `List[float]`, *optional*, defaults to 1.0):
                The percentage of total steps at which the ControlNet stops applying.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list with the generated images and the
                second element is a list of `bool`s indicating whether the corresponding generated image contains
                "not-safe-for-work" (nsfw) content.
        """
        # controlnet = self.controlnet._orig_mod if is_compiled_module(self.controlnet) else self.controlnet

        # # align format for control guidance
        # if not isinstance(control_guidance_start, list) and isinstance(control_guidance_end, list):
        #     control_guidance_start = len(control_guidance_end) * [control_guidance_start]
        # elif not isinstance(control_guidance_end, list) and isinstance(control_guidance_start, list):
        #     control_guidance_end = len(control_guidance_start) * [control_guidance_end]
        # elif not isinstance(control_guidance_start, list) and not isinstance(control_guidance_end, list):
        #     mult = len(controlnet.nets) if isinstance(controlnet, MultiControlNetModel) else 1
        #     control_guidance_start, control_guidance_end = mult * [control_guidance_start], mult * [
        #         control_guidance_end
        #     ]

        # 1. Check inputs. Raise error if not correct
        # self.check_inputs(
        #     prompt,
        #     control_image,
        #     callback_steps,
        #     negative_prompt,
        #     prompt_embeds,
        #     negative_prompt_embeds,
        #     controlnet_conditioning_scale,
        #     control_guidance_start,
        #     control_guidance_end,
        # )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # if isinstance(controlnet, MultiControlNetModel) and isinstance(controlnet_conditioning_scale, float):
        #     controlnet_conditioning_scale = [controlnet_conditioning_scale] * len(controlnet.nets)

        # global_pool_conditions = (
        #     controlnet.config.global_pool_conditions
        #     if isinstance(controlnet, ControlNetModel)
        #     else controlnet.nets[0].config.global_pool_conditions
        # )
        # guess_mode = guess_mode or global_pool_conditions

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
            clip_skip=clip_skip,
        )
        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        # 4. Prepare image
        #image = preprocess(image)
        image = self.image_processor.preprocess(image, height=height, width=width).to(dtype=torch.float32)
        # 5. Prepare controlnet_conditioning_image
        # if isinstance(controlnet, ControlNetModel):
        #     control_image = self.prepare_control_image(
        #         image=control_image,
        #         width=width,
        #         height=height,
        #         batch_size=batch_size * num_images_per_prompt,
        #         num_images_per_prompt=num_images_per_prompt,
        #         device=device,
        #         dtype=controlnet.dtype,
        #         do_classifier_free_guidance=do_classifier_free_guidance,
        #         guess_mode=guess_mode,
        #     )
        # elif isinstance(controlnet, MultiControlNetModel):
        #     control_images = []

        #     for control_image_ in control_image:
        #         control_image_ = self.prepare_control_image(
        #             image=control_image_,
        #             width=width,
        #             height=height,
        #             batch_size=batch_size * num_images_per_prompt,
        #             num_images_per_prompt=num_images_per_prompt,
        #             device=device,
        #             dtype=controlnet.dtype,
        #             do_classifier_free_guidance=do_classifier_free_guidance,
        #             guess_mode=guess_mode,
        #         )

        #         control_images.append(control_image_)

        #     control_image = control_images
        # else:
        #     assert False

        # 5. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength, device)
        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)
        
        # 6. Prepare latent variables
        latents_img, latents_img_ori = self.prepare_latents(
                image,
                latent_timestep,
                batch_size,
                num_images_per_prompt,
                prompt_embeds.dtype,
                device,
                generator,
                latents,
                mask_img=None,
                latent_blend_kernel=latent_blend_kernel,
            )

        # 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7.1 Create tensor stating which controlnets to keep
        # controlnet_keep = []
        # for i in range(len(timesteps)):
        #     keeps = [
        #         1.0 - float(i / len(timesteps) < s or (i + 1) / len(timesteps) > e)
        #         for s, e in zip(control_guidance_start, control_guidance_end)
        #     ]
        #     controlnet_keep.append(keeps[0] if isinstance(controlnet, ControlNetModel) else keeps)

        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order

        latents_img = latents_img_ori
        start_latents = latents_img_ori
        latents_list = [latents_img_ori]
        noise_list = [torch.zeros_like(latents_img_ori)]

        #with self.progress_bar(total=num_inference_steps) as progress_bar:
        with tqdm(total=num_inference_steps, desc=desc) as progress_bar:
            for i, t in enumerate(reversed(timesteps)):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents_img] * 2) if do_classifier_free_guidance else latents_img
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # controlnet(s) inference
                # if guess_mode and do_classifier_free_guidance:
                #     # Infer ControlNet only for the conditional batch.
                #     control_model_input = latents_img
                #     control_model_input = self.scheduler.scale_model_input(control_model_input, t)
                #     controlnet_prompt_embeds = prompt_embeds.chunk(2)[1]
                # else:
                #     control_model_input = latent_model_input
                #     controlnet_prompt_embeds = prompt_embeds

                # if isinstance(controlnet_keep[i], list):
                #     cond_scale = [c * s for c, s in zip(controlnet_conditioning_scale, controlnet_keep[i])]
                # else:
                #     controlnet_cond_scale = controlnet_conditioning_scale
                #     if isinstance(controlnet_cond_scale, list):
                #         controlnet_cond_scale = controlnet_cond_scale[0]
                #     cond_scale = controlnet_cond_scale * controlnet_keep[i]

                # down_block_res_samples, mid_block_res_sample = self.controlnet(
                #     control_model_input,
                #     t,
                #     encoder_hidden_states=controlnet_prompt_embeds,
                #     controlnet_cond=control_image,
                #     conditioning_scale=cond_scale,
                #     guess_mode=guess_mode,
                #     return_dict=False,
                # )

                # if guess_mode and do_classifier_free_guidance:
                #     # Infered ControlNet only for the conditional batch.
                #     # To apply the output of ControlNet to both the unconditional and conditional batches,
                #     # add 0 to the unconditional batch to keep it unchanged.
                #     down_block_res_samples = [torch.cat([torch.zeros_like(d), d]) for d in down_block_res_samples]
                #     mid_block_res_sample = torch.cat([torch.zeros_like(mid_block_res_sample), mid_block_res_sample])

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                    # down_block_additional_residuals=down_block_res_samples,
                    # mid_block_additional_residual=mid_block_res_sample,
                    return_dict=False,
                )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                #latents_img = self.scheduler.step(noise_pred, t, latents_img, **extra_step_kwargs, return_dict=False)[0]
                
                latents_img = self.next_step(noise_pred, t, latents_img)
                
                latents_list.append(latents_img)
                noise_list.append(noise_pred)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

        # If we do sequential model offloading, let's offload unet and controlnet
        # manually for max memory savings
        # if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
        #     self.unet.to("cpu")
        #     self.controlnet.to("cpu")
        #     torch.cuda.empty_cache()

        if return_intermediates:
            return latents_img, latents_list, noise_list
        # Offload all models
        # self.maybe_free_model_hooks()

        # if not return_dict:
        #     return (image, has_nsfw_concept)

        return latents_img, start_latents
    
    @torch.no_grad()
    def vae_invert(
        self,
        prompt: Union[str, List[str]] = None,
        image: Union[torch.FloatTensor, PIL.Image.Image] = None,
        control_image: Union[torch.FloatTensor, PIL.Image.Image] = None,
        mask_image: Union[torch.FloatTensor, PIL.Image.Image] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        strength: float = 0.8,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_conditioning_scale: Union[float, List[float]] = 0.8,
        guess_mode: bool = False,
        control_guidance_start: Union[float, List[float]] = 0.0,
        control_guidance_end: Union[float, List[float]] = 1.0,
        clip_skip: Optional[int] = None,
        latent_blend_kernel: Optional[int] = -1,
        return_intermediates : bool = False,
        inpainting_strength: Optional[float] = 1,
        mask_blend_kernel: Optional[int] = -1,
        return_vae_list: bool = False,
        desc: Optional[str] = None,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            image (`torch.FloatTensor`, `PIL.Image.Image`, `np.ndarray`, `List[torch.FloatTensor]`, `List[PIL.Image.Image]`, `List[np.ndarray]`,:
                    `List[List[torch.FloatTensor]]`, `List[List[np.ndarray]]` or `List[List[PIL.Image.Image]]`):
                The initial image to be used as the starting point for the image generation process. Can also accept
                image latents as `image`, and if passing latents directly they are not encoded again.
            control_image (`torch.FloatTensor`, `PIL.Image.Image`, `np.ndarray`, `List[torch.FloatTensor]`, `List[PIL.Image.Image]`, `List[np.ndarray]`,:
                    `List[List[torch.FloatTensor]]`, `List[List[np.ndarray]]` or `List[List[PIL.Image.Image]]`):
                The ControlNet input condition to provide guidance to the `unet` for generation. If the type is
                specified as `torch.FloatTensor`, it is passed to ControlNet as is. `PIL.Image.Image` can also be
                accepted as an image. The dimensions of the output image defaults to `image`'s dimensions. If height
                and/or width are passed, `image` is resized accordingly. If multiple ControlNets are specified in
                `init`, images must be passed as a list such that each element of the list can be correctly batched for
                input to a single ControlNet.
            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that calls every `callback_steps` steps during inference. The function is called with the
                following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function is called. If not specified, the callback is called at
                every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            controlnet_conditioning_scale (`float` or `List[float]`, *optional*, defaults to 1.0):
                The outputs of the ControlNet are multiplied by `controlnet_conditioning_scale` before they are added
                to the residual in the original `unet`. If multiple ControlNets are specified in `init`, you can set
                the corresponding scale as a list.
            guess_mode (`bool`, *optional*, defaults to `False`):
                The ControlNet encoder tries to recognize the content of the input image even if you remove all
                prompts. A `guidance_scale` value between 3.0 and 5.0 is recommended.
            control_guidance_start (`float` or `List[float]`, *optional*, defaults to 0.0):
                The percentage of total steps at which the ControlNet starts applying.
            control_guidance_end (`float` or `List[float]`, *optional*, defaults to 1.0):
                The percentage of total steps at which the ControlNet stops applying.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list with the generated images and the
                second element is a list of `bool`s indicating whether the corresponding generated image contains
                "not-safe-for-work" (nsfw) content.
        """
        # controlnet = self.controlnet._orig_mod if is_compiled_module(self.controlnet) else self.controlnet

        # # align format for control guidance
        # if not isinstance(control_guidance_start, list) and isinstance(control_guidance_end, list):
        #     control_guidance_start = len(control_guidance_end) * [control_guidance_start]
        # elif not isinstance(control_guidance_end, list) and isinstance(control_guidance_start, list):
        #     control_guidance_end = len(control_guidance_start) * [control_guidance_end]
        # elif not isinstance(control_guidance_start, list) and not isinstance(control_guidance_end, list):
        #     mult = len(controlnet.nets) if isinstance(controlnet, MultiControlNetModel) else 1
        #     control_guidance_start, control_guidance_end = mult * [control_guidance_start], mult * [
        #         control_guidance_end
        #     ]

        # 1. Check inputs. Raise error if not correct
        # self.check_inputs(
        #     prompt,
        #     control_image,
        #     callback_steps,
        #     negative_prompt,
        #     prompt_embeds,
        #     negative_prompt_embeds,
        #     controlnet_conditioning_scale,
        #     control_guidance_start,
        #     control_guidance_end,
        # )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # if isinstance(controlnet, MultiControlNetModel) and isinstance(controlnet_conditioning_scale, float):
        #     controlnet_conditioning_scale = [controlnet_conditioning_scale] * len(controlnet.nets)

        # global_pool_conditions = (
        #     controlnet.config.global_pool_conditions
        #     if isinstance(controlnet, ControlNetModel)
        #     else controlnet.nets[0].config.global_pool_conditions
        # )
        # guess_mode = guess_mode or global_pool_conditions

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
            clip_skip=clip_skip,
        )
        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        # 4. Prepare image
        #image = preprocess(image)
        image = self.image_processor.preprocess(image, height=height, width=width).to(dtype=torch.float32)
        # 5. Prepare controlnet_conditioning_image
        # if isinstance(controlnet, ControlNetModel):
        #     control_image = self.prepare_control_image(
        #         image=control_image,
        #         width=width,
        #         height=height,
        #         batch_size=batch_size * num_images_per_prompt,
        #         num_images_per_prompt=num_images_per_prompt,
        #         device=device,
        #         dtype=controlnet.dtype,
        #         do_classifier_free_guidance=do_classifier_free_guidance,
        #         guess_mode=guess_mode,
        #     )
        # elif isinstance(controlnet, MultiControlNetModel):
        #     control_images = []

        #     for control_image_ in control_image:
        #         control_image_ = self.prepare_control_image(
        #             image=control_image_,
        #             width=width,
        #             height=height,
        #             batch_size=batch_size * num_images_per_prompt,
        #             num_images_per_prompt=num_images_per_prompt,
        #             device=device,
        #             dtype=controlnet.dtype,
        #             do_classifier_free_guidance=do_classifier_free_guidance,
        #             guess_mode=guess_mode,
        #         )

        #         control_images.append(control_image_)

        #     control_image = control_images
        # else:
        #     assert False

        # 5. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength, device)
        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)
        
        # 6. Prepare latent variables
        latents_img, latents_img_ori = self.prepare_latents(
                image,
                latent_timestep,
                batch_size,
                num_images_per_prompt,
                prompt_embeds.dtype,
                device,
                generator,
                latents,
                mask_img=None,
                latent_blend_kernel=latent_blend_kernel,
            )

        # 7. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)
        
        if mask_image is not None:
            if not isinstance(mask_image, torch.FloatTensor):
                vae_mask = preprocess_mask(
                    mask_image, 1, mask_blend_kernel=mask_blend_kernel
                )
            mask = mask_image.to(device=self.device, dtype=latents_img.dtype)
            #mask_cat = torch.cat([mask] * batch_size * num_images_per_prompt)

        # 7.1 Create tensor stating which controlnets to keep
        # controlnet_keep = []
        # for i in range(len(timesteps)):
        #     keeps = [
        #         1.0 - float(i / len(timesteps) < s or (i + 1) / len(timesteps) > e)
        #         for s, e in zip(control_guidance_start, control_guidance_end)
        #     ]
        #     controlnet_keep.append(keeps[0] if isinstance(controlnet, ControlNetModel) else keeps)

        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order

        latents_img = latents_img_ori
        start_latents = latents_img_ori
        latents_list = [latents_img_ori]
        noise_list = [torch.zeros_like(latents_img_ori)]

        #with self.progress_bar(total=num_inference_steps) as progress_bar:
        with tqdm(total=num_inference_steps, desc=desc) as progress_bar:
            for i, t in enumerate(reversed(timesteps)):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents_img] * 2) if do_classifier_free_guidance else latents_img
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # controlnet(s) inference
                # if guess_mode and do_classifier_free_guidance:
                #     # Infer ControlNet only for the conditional batch.
                #     control_model_input = latents_img
                #     control_model_input = self.scheduler.scale_model_input(control_model_input, t)
                #     controlnet_prompt_embeds = prompt_embeds.chunk(2)[1]
                # else:
                #     control_model_input = latent_model_input
                #     controlnet_prompt_embeds = prompt_embeds

                # if isinstance(controlnet_keep[i], list):
                #     cond_scale = [c * s for c, s in zip(controlnet_conditioning_scale, controlnet_keep[i])]
                # else:
                #     controlnet_cond_scale = controlnet_conditioning_scale
                #     if isinstance(controlnet_cond_scale, list):
                #         controlnet_cond_scale = controlnet_cond_scale[0]
                #     cond_scale = controlnet_cond_scale * controlnet_keep[i]

                # down_block_res_samples, mid_block_res_sample = self.controlnet(
                #     control_model_input,
                #     t,
                #     encoder_hidden_states=controlnet_prompt_embeds,
                #     controlnet_cond=control_image,
                #     conditioning_scale=cond_scale,
                #     guess_mode=guess_mode,
                #     return_dict=False,
                # )

                # if guess_mode and do_classifier_free_guidance:
                #     # Infered ControlNet only for the conditional batch.
                #     # To apply the output of ControlNet to both the unconditional and conditional batches,
                #     # add 0 to the unconditional batch to keep it unchanged.
                #     down_block_res_samples = [torch.cat([torch.zeros_like(d), d]) for d in down_block_res_samples]
                #     mid_block_res_sample = torch.cat([torch.zeros_like(mid_block_res_sample), mid_block_res_sample])

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                    # down_block_additional_residuals=down_block_res_samples,
                    # mid_block_additional_residual=mid_block_res_sample,
                    return_dict=False,
                )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                #latents_img = self.scheduler.step(noise_pred, t, latents_img, **extra_step_kwargs, return_dict=False)[0]
                
                latents_img = self.next_step(noise_pred, t, latents_img)
                
                latents_list.append(latents_img)
                noise_list.append(noise_pred)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

        # If we do sequential model offloading, let's offload unet and controlnet
        # manually for max memory savings
        # if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
        #     self.unet.to("cpu")
        #     self.controlnet.to("cpu")
        #     torch.cuda.empty_cache()
        # vae_list = None
        # if return_intermediates and return_vae_list:
        #     image, vae_list= self.vae.decode(latents_img / self.vae.config.scaling_factor, return_dict=False, mask=vae_mask, return_list=True)[0]
        #     return latents_img, latents_list, noise_list, vae_list
        
        if return_intermediates:
            return latents_img, latents_list, noise_list
        # Offload all models
        # self.maybe_free_model_hooks()

        # if not return_dict:
        #     return (image, has_nsfw_concept)

        return latents_img, start_latents