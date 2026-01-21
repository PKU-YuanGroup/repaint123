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
from typing import Callable, List, Optional, Union

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
from diffusers.models import AutoencoderKL, UNet2DConditionModel
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

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

from .mask_options import *
from inpaint.view_generation.depth_supervised_inpainting_pipeline import StableDiffusionDepth2ImgInpaintingPipeline

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


class MasaCtrlPipeline(StableDiffusionDepth2ImgInpaintingPipeline):
    
    def prepare_depth_map(
        self, image, depth_map, batch_size, do_classifier_free_guidance, dtype, device
    ):
        if isinstance(image, PIL.Image.Image):
            image = [image]
        else:
            image = [img for img in image]

        if depth_map is None:
            pixel_values = self.feature_extractor(
                images=image, return_tensors="pt"
            ).pixel_values
            pixel_values = pixel_values.to(device=device)
            # The DPT-Hybrid model uses batch-norm layers which are not compatible with fp16.
            # So we use `torch.autocast` here for half precision inference.
            context_manger = (
                torch.autocast("cuda", dtype=dtype)
                if device.type == "cuda"
                else contextlib.nullcontext()
            )
            with context_manger:
                depth_map = self.depth_estimator(pixel_values).predicted_depth
        else:
            depth_map = depth_map.to(device=device, dtype=dtype)  # [1, 64, 64]

        depth_map = depth_map.cpu().numpy()
        depth_map = skimage.measure.block_reduce(
            depth_map, (1, self.vae_scale_factor, self.vae_scale_factor), custom_mean
        )

        depth_map = torch.tensor(depth_map).to(device=device, dtype=dtype)[None,]
        if depth_map.shape[1] == 2 :
            depth_map = torch.reshape(depth_map, (2, 1, 64, 64))
        depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
        depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
        depth_map = 2.0 * (depth_map - depth_min) / (depth_max - depth_min) - 1.0
        depth_map = depth_map.to(dtype)

        # duplicate mask and masked_image_latents for each generation per prompt, using mps friendly method
        if depth_map.shape[0] < batch_size:
            depth_map = depth_map.repeat(batch_size, 1, 1, 1)

        depth_map = (
            torch.cat([depth_map] * 2) if do_classifier_free_guidance else depth_map
        )
        return depth_map
    
    @torch.no_grad()
    def dynamic_refine(self, t, mask):
        step_f = torch.linspace(0, 1.0, t+2, dtype=mask.dtype).to(mask.device)
        step_mask = torch.searchsorted(step_f, mask)
        step_mask = step_mask -1 
        return step_mask

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str], torch.FloatTensor],
        image: Union[torch.FloatTensor, PIL.Image.Image],
        latent_code: Union[torch.FloatTensor, PIL.Image.Image] = None,
        latent_list: Union[torch.FloatTensor, PIL.Image.Image] = None,
        mask_image: Union[torch.FloatTensor, PIL.Image.Image] = None,
        depth_map: Optional[torch.FloatTensor] = None,
        strength: float = 0.8,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: Optional[float] = 0.0,
        add_predicted_noise: Optional[bool] = False,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        latents: Optional[torch.FloatTensor] = None,
        inpainting_strength: Optional[float] = 1,
        mask_blend_kernel: Optional[int] = -1,
        latent_blend_kernel: Optional[int] = -1,
        desc: Optional[str] = None,
        depth_invert: bool = False,
        dynamic_mask : bool =False,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide the image generation.
            image (`torch.FloatTensor` or `PIL.Image.Image`):
                `Image`, or tensor representing an image batch, that will be used as the starting point for the
                process.
            strength (`float`, *optional*, defaults to 0.8):
                Conceptually, indicates how much to transform the reference `image`. Must be between 0 and 1. `image`
                will be used as a starting point, adding more noise to it the larger the `strength`. The number of
                denoising steps depends on the amount of noise initially added. When `strength` is 1, added noise will
                be maximum and the denoising process will run for the full number of iterations specified in
                `num_inference_steps`. A value of 1, therefore, essentially ignores `image`.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference. This parameter will be modulated by `strength`.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """
        # 1. Check inputs
        self.check_inputs(prompt, strength, callback_steps)

        # 2. Define call parameters
        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        if isinstance(prompt, torch.Tensor):
            text_embeddings = prompt
        else :
            text_embeddings = self._encode_prompt(
                prompt,
                device,
                num_images_per_prompt,
                do_classifier_free_guidance,
                negative_prompt,
            )

        # 4. Preprocess image
        depth_mask = self.prepare_depth_map(
            image,
            depth_map,
            batch_size * num_images_per_prompt,
            do_classifier_free_guidance,
            text_embeddings.dtype,
            device,
        )

        # 5. Prepare depth mask
        image = preprocess(image)

        # 6. set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        num_total_steps = num_inference_steps
        timesteps, num_inference_steps = self.get_timesteps(
            num_inference_steps, strength, device
        )
        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)

        # 7. Prepare latent variables
        mask_img = None
        if not isinstance(mask_image, torch.FloatTensor) and latent_blend_kernel > 0:
            mask_img = preprocess_mask(
                mask_image, self.vae_scale_factor, mask_blend_kernel=-1
            )
            mask_img = (
                PIL.Image.fromarray(
                    (mask_img[0, 0].cpu().numpy() * 255).astype(np.uint8)
                )
                .convert("1")
                .convert("L")
            )
        
        latents_img, latents_img_ori = self.prepare_latents(
                image,
                latent_timestep,
                batch_size,
                num_images_per_prompt,
                text_embeddings.dtype,
                device,
                generator,
                latents,
                mask_img=mask_img,
                latent_blend_kernel=latent_blend_kernel,
            )
        
        if latent_code is not None:
            if isinstance(latent_code, list):
                latents_img = torch.cat(latent_code, dim=0)
            else :
                latents_img = latent_code

        # 7. Prepare mask latent

        if not isinstance(mask_image, torch.FloatTensor):
            mask_image = preprocess_mask(
                mask_image, self.vae_scale_factor, mask_blend_kernel=mask_blend_kernel
            )
        mask = mask_image.to(device=self.device, dtype=latents_img.dtype)
        if dynamic_mask is True:
            mask = self.dynamic_refine(num_total_steps, mask)
        mask = torch.cat([mask] * batch_size * num_images_per_prompt)

        # 8. Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 9. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with tqdm(total=num_inference_steps, desc=desc) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = (
                    torch.cat([latents_img] * 2)
                    if do_classifier_free_guidance
                    else latents_img
                )
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t
                )
                latent_model_input = torch.cat([latent_model_input, depth_mask], dim=1)

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input, t, encoder_hidden_states=text_embeddings
                ).sample

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                # compute the previous noisy sample x_t -> x_t-1
                latents_img = self.scheduler.step(
                    noise_pred, t, latents_img, **extra_step_kwargs
                ).prev_sample

                if latent_list is not None:
                    latents_img[:1, :, :, :] = latent_list[0][-i-2]
                    if t > 1000 * (1 - inpainting_strength):
                        if dynamic_mask is True:
                            mask_t = (mask > i + num_total_steps - num_inference_steps).float()
                        else :
                            mask_t = mask 
                        latents_img[1:2, :, :, :]= (latent_list[1][-i-2] * mask_t[1]) + (
                            latents_img[1:2, :, :, :] * (1 - mask_t[1])
                        )
                else :
                    if add_predicted_noise:
                        init_latents_proper = self.scheduler.add_noise(
                            latents_img_ori, noise_pred_uncond, torch.tensor([t])
                        )
                    else:
                        init_latents_proper = self.scheduler.add_noise(
                            latents_img_ori, latents, torch.tensor([t])
                        )
                    latents_img[:1, :, :, :]= init_latents_proper[ :1, :, :, :]
                    if t > 1000 * (1 - inpainting_strength):
                        latents_img[1:2, :, :, :]= (init_latents_proper[1:2, :, :, :] * mask[1]) + (
                            latents_img[1:2, :, :, :] * (1 - mask[1])
                        )

                # if t > 1000 * (1 - inpainting_strength):
                #     if latent_list is not None:
                #         #latents_img[:1, :, :, :] = latent_list[0][-i-2]
                #         latents_img[1:2, :, :, :]= (latent_list[1][-i-2] * mask[1]) + (
                #             latents_img[1:2, :, :, :] * (1 - mask[1])
                #         )
                #     else :
                #         if add_predicted_noise:
                #             init_latents_proper = self.scheduler.add_noise(
                #                 latents_img_ori, noise_pred_uncond, torch.tensor([t])
                #             )
                #         else:
                #             init_latents_proper = self.scheduler.add_noise(
                #                 latents_img_ori, latents, torch.tensor([t])
                #             )
                        
                #         latents_img[:1, :, :, :]= init_latents_proper[ :1, :, :, :]
                #         latents_img[1:2, :, :, :]= (init_latents_proper[1:2, :, :, :] * mask[1]) + (
                #             latents_img[1:2, :, :, :] * (1 - mask[1])
                #         )

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents_img)

        # 10. Post-processing
        image = self.decode_latents(latents_img)

        # 11. Convert to PIL
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

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
    
    @torch.no_grad()
    def depth_invert(
        self,
        prompt: Union[str, List[str], torch.FloatTensor],
        image: Union[torch.FloatTensor, PIL.Image.Image],
        mask_image: Union[torch.FloatTensor, PIL.Image.Image] = None,
        depth_map: Optional[torch.FloatTensor] = None,
        strength: float = 0.8,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: Optional[float] = 0.0,
        add_predicted_noise: Optional[bool] = False,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        latents: Optional[torch.FloatTensor] = None,
        inpainting_strength: Optional[float] = 1,
        mask_blend_kernel: Optional[int] = -1,
        latent_blend_kernel: Optional[int] = -1,
        desc: Optional[str] = None,
        return_intermediates: bool = False,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide the image generation.
            image (`torch.FloatTensor` or `PIL.Image.Image`):
                `Image`, or tensor representing an image batch, that will be used as the starting point for the
                process.
            strength (`float`, *optional*, defaults to 0.8):
                Conceptually, indicates how much to transform the reference `image`. Must be between 0 and 1. `image`
                will be used as a starting point, adding more noise to it the larger the `strength`. The number of
                denoising steps depends on the amount of noise initially added. When `strength` is 1, added noise will
                be maximum and the denoising process will run for the full number of iterations specified in
                `num_inference_steps`. A value of 1, therefore, essentially ignores `image`.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference. This parameter will be modulated by `strength`.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """
        # 1. Check inputs
        self.check_inputs(prompt, strength, callback_steps)

        # 2. Define call parameters
        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        if isinstance(prompt, torch.Tensor):
            text_embeddings = prompt
        else :
            text_embeddings = self._encode_prompt(
                prompt,
                device,
                num_images_per_prompt,
                do_classifier_free_guidance,
                negative_prompt,
            )

        # 4. Preprocess image
        depth_mask = self.prepare_depth_map(
            image,
            depth_map,
            batch_size * num_images_per_prompt,
            do_classifier_free_guidance,
            text_embeddings.dtype,
            device,
        )

        # 5. Prepare depth mask
        image = preprocess(image)

        # 6. set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps, num_inference_steps = self.get_timesteps(
            num_inference_steps, strength, device
        )
        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)

        # 7. Prepare latent variables
        mask_img = None
        if not isinstance(mask_image, torch.FloatTensor) and latent_blend_kernel > 0:
            mask_img = preprocess_mask(
                mask_image, self.vae_scale_factor, mask_blend_kernel=-1
            )
            mask_img = (
                PIL.Image.fromarray(
                    (mask_img[0, 0].cpu().numpy() * 255).astype(np.uint8)
                )
                .convert("1")
                .convert("L")
            )

        latents_img, latents_img_ori = self.prepare_latents(
            image,
            latent_timestep,
            batch_size,
            num_images_per_prompt,
            text_embeddings.dtype,
            device,
            generator,
            latents,
            mask_img=mask_img,
            latent_blend_kernel=latent_blend_kernel,
        )

        # 7. Prepare mask latent

        # if not isinstance(mask_image, torch.FloatTensor):
        #     mask_image = preprocess_mask(
        #         mask_image, self.vae_scale_factor, mask_blend_kernel=mask_blend_kernel
        #     )
        # mask = mask_image.to(device=self.device, dtype=latents_img.dtype)
        # mask = torch.cat([mask] * batch_size * num_images_per_prompt)

        # 8. Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 9. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order

        latents_img = latents_img_ori
        start_latents = latents_img_ori
        latents_list = [latents_img_ori]
        with tqdm(total=num_inference_steps, desc=desc) as progress_bar:
            for i, t in enumerate(reversed(timesteps)):
                # expand the latents if we are doing classifier free guidance
                if i==0:
                    t = torch.ones_like(t)
                else :
                    t = (reversed(timesteps))[i-1]

                latent_model_input = (
                    torch.cat([latents_img] * 2)
                    if do_classifier_free_guidance
                    else latents_img
                )
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t
                )
                latent_model_input = torch.cat([latent_model_input, depth_mask], dim=1)

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input, t, encoder_hidden_states=text_embeddings
                ).sample

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                # compute the previous noisy sample x_t -> x_t-1
                # latents_img = self.scheduler.step(
                #     noise_pred, t, latents_img, **extra_step_kwargs
                # ).prev_sample
                #latents_img = self.next_step(noise_pred, t, latents_img)
                
                if i == 0:
                    latents_img = self.next_step(noise_pred, t, latents_img)
                else:
                    latents_img = self.next_step(noise_pred, t + self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, latents_img)

                latents_list.append(latents_img)
                # call the callback, if provided
                # if i == len(timesteps) - 1 or (
                #     (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                # ):
                #     progress_bar.update()
                #     if callback is not None and i % callback_steps == 0:
                #         callback(i, t, latents_img)

        if return_intermediates:
            return latents_img, latents_list
        
        return latents_img, start_latents
        # 10. Post-processing
        #image = self.decode_latents(latents_img)
        
        # 11. Convert to PIL
        # if output_type == "pil":
        #     image = self.numpy_to_pil(image)

        # if not return_dict:
        #     return (image,)

        #return ImagePipelineOutput(images=image[-1:])