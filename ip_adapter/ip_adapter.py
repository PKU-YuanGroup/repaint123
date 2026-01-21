import torch
from diffusers.pipelines.controlnet import MultiControlNetModel
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
from PIL import Image

if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
    from .attention_processor import IPAttnProcessor2_0 as IPAttnProcessor, AttnProcessor2_0 as AttnProcessor, CNAttnProcessor2_0 as CNAttnProcessor
else:
    from .attention_processor import IPAttnProcessor, AttnProcessor, CNAttnProcessor
from .resampler import Resampler

class ImageProjModel(torch.nn.Module):
    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=4):
        super().__init__()
        
        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = torch.nn.Linear(clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim)
        self.norm = torch.nn.LayerNorm(cross_attention_dim)
        
    def forward(self, image_embeds):
        embeds = image_embeds
        clip_extra_context_tokens = self.proj(embeds).reshape(-1, self.clip_extra_context_tokens, self.cross_attention_dim)
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        return clip_extra_context_tokens

class IPAdapter:
    def __init__(self, pipe, ipadapter_ckpt_path, image_encoder_path,scale=1, device="cuda", dtype=torch.float32, resample=Image.Resampling.LANCZOS):
        self.pipe = pipe
        self.device = device
        self.dtype = dtype
        self.scale = scale
        
        # load ip adapter model
        ipadapter_model = torch.load(ipadapter_ckpt_path, map_location="cpu")

        # detect features
        self.is_plus = "latents" in ipadapter_model["image_proj"]
        self.output_cross_attention_dim = ipadapter_model["ip_adapter"]["1.to_k_ip.weight"].shape[1]
        self.is_sdxl = self.output_cross_attention_dim == 2048
        self.cross_attention_dim = 1280 if self.is_plus and self.is_sdxl else self.output_cross_attention_dim
        self.heads = 20 if self.is_sdxl and self.is_plus else 12
        self.num_tokens = 16 if self.is_plus else 4

        # set image encoder
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(image_encoder_path).to(self.device, dtype=self.dtype)
        self.clip_image_processor = CLIPImageProcessor(resample=resample)

        # set IPAdapter
        self.set_ip_adapter()
        self.image_proj_model = self.init_proj() if not self.is_plus else self.init_proj_plus()
        self.image_proj_model.load_state_dict(ipadapter_model["image_proj"])
        ip_layers = torch.nn.ModuleList(self.pipe.unet.attn_processors.values())
        ip_layers.load_state_dict(ipadapter_model["ip_adapter"])
        
    def init_proj(self):
        image_proj_model = ImageProjModel(
            cross_attention_dim=self.cross_attention_dim,
            clip_embeddings_dim=self.image_encoder.config.projection_dim,
            clip_extra_context_tokens=self.num_tokens,
        ).to(self.device, dtype=self.dtype)
        return image_proj_model
    
    def init_proj_plus(self):
        image_proj_model = Resampler(
            dim=self.cross_attention_dim,
            depth=4,
            dim_head=64,
            heads=self.heads,
            num_queries=self.num_tokens,
            embedding_dim=self.image_encoder.config.hidden_size,
            output_dim=self.output_cross_attention_dim,
            ff_mult=4
        ).to(self.device, dtype=torch.float32)
        return image_proj_model

    def set_ip_adapter(self):
        unet = self.pipe.unet
        attn_procs = {}
        for name in unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            if cross_attention_dim is None:
                attn_procs[name] = AttnProcessor()
            else:
                attn_procs[name] = IPAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim,scale=self.scale).to(self.device, dtype=self.dtype)
        unet.set_attn_processor(attn_procs)
        if hasattr(self.pipe, "controlnet"):
            if isinstance(self.pipe.controlnet, MultiControlNetModel):
                for controlnet in self.pipe.controlnet.nets:
                    controlnet.set_attn_processor(CNAttnProcessor())
            else:
                self.pipe.controlnet.set_attn_processor(CNAttnProcessor())
    
    @torch.inference_mode()
    def get_image_embeds(self, images, negative_images=None):
        clip_image = self.clip_image_processor(images=images, return_tensors="pt").pixel_values
        clip_image = clip_image.to(self.device, dtype=torch.float32)

        if not self.is_plus:
            clip_image_embeds = self.image_encoder(clip_image).image_embeds
            image_prompt_embeds = self.image_proj_model(clip_image_embeds)
            if negative_images is not None:
                negative_clip_image = self.clip_image_processor(images=negative_images, return_tensors="pt").pixel_values
                negative_clip_image = negative_clip_image.to(self.device, dtype=torch.float32)
                negative_image_prompt_embeds = self.image_encoder(negative_clip_image).image_embeds
            else:
                negative_image_prompt_embeds = torch.zeros_like(clip_image_embeds)
            negative_image_prompt_embeds = self.image_proj_model(negative_image_prompt_embeds)
        else:
            clip_image_embeds = self.image_encoder(clip_image, output_hidden_states=True).hidden_states[-2]
            image_prompt_embeds = self.image_proj_model(clip_image_embeds)
            if negative_images is not None:
                negative_clip_image = self.clip_image_processor(images=negative_images, return_tensors="pt").pixel_values
                negative_clip_image = negative_clip_image.to(self.device, dtype=torch.float32)
                negative_clip_image_embeds = self.image_encoder(negative_clip_image, output_hidden_states=True).hidden_states[-2]
            else:
                negative_clip_image_embeds = self.image_encoder(torch.zeros_like(clip_image), output_hidden_states=True).hidden_states[-2]
            negative_image_prompt_embeds = self.image_proj_model(negative_clip_image_embeds)
        
        num_tokens = image_prompt_embeds.shape[0] * self.num_tokens
        self.set_tokens(num_tokens)

        return image_prompt_embeds, negative_image_prompt_embeds

    @torch.inference_mode()
    def get_prompt_embeds(self, images, negative_images=None, prompt=None, negative_prompt=None, weight=[]):
        prompt_embeds, negative_prompt_embeds = self.get_image_embeds(images, negative_images=negative_images)

        if any(e != 1.0 for e in weight):
            weight = torch.tensor(weight).unsqueeze(-1).unsqueeze(-1)
            weight = weight.to(self.device)
            prompt_embeds = prompt_embeds * weight

        if prompt_embeds.shape[0] > 1:
            prompt_embeds = torch.cat(prompt_embeds.chunk(prompt_embeds.shape[0]), dim=1)
        if negative_prompt_embeds.shape[0] > 1:
            negative_prompt_embeds = torch.cat(negative_prompt_embeds.chunk(negative_prompt_embeds.shape[0]), dim=1)

        text_embeds = (None, None, None, None)
        if prompt is not None:
            text_embeds = self.pipe.encode_prompt(
                prompt,
                negative_prompt=negative_prompt,
                device=self.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True
            )
            prompt_embeds = torch.cat((text_embeds[0], prompt_embeds), dim=1)
            negative_prompt_embeds = torch.cat((text_embeds[1], negative_prompt_embeds), dim=1)

        output = (prompt_embeds, negative_prompt_embeds)

        if self.is_sdxl:
            output += (text_embeds[2], text_embeds[3])
        
        return output

    def set_scale(self, scale):
        for attn_processor in self.pipe.unet.attn_processors.values():
            if isinstance(attn_processor, IPAttnProcessor):
                attn_processor.scale = scale
    
    def set_tokens(self, num_tokens):
        for attn_processor in self.pipe.unet.attn_processors.values():
            if isinstance(attn_processor, IPAttnProcessor):
                attn_processor.num_tokens = num_tokens
