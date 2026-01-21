import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Union, Tuple, List, Callable, Dict

from torchvision.utils import save_image
from einops import rearrange, repeat


class AttentionBase:
    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0
        
        # self.reset_hook()

    def after_step(self):
        pass

    def __call__(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        out = self.forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            # after step
            self.after_step()
        return out

    def forward(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        # if(self.hook and not is_cross):
        #     self.kv_cache[self.cur_step][self.cur_att_layer]['k'] = k
        #     self.kv_cache[self.cur_step][self.cur_att_layer]['v'] = v
        out = torch.einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=num_heads)
        return out

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0
        
    # def reset_hook(self):
    #     self.hook = False
    #     self.kv_cache = {}
    #     for step in range(1000):
    #         self.kv_cache[step] = {}
    #         for layer in range(32):
    #             self.kv_cache[step][layer] = {}
    #             self.kv_cache[step][layer]['k'] = None
    #             self.kv_cache[step][layer]['v'] = None


class AttentionStore(AttentionBase):
    def __init__(self, res=[32], min_step=0, max_step=1000):
        super().__init__()
        self.res = res
        self.min_step = min_step
        self.max_step = max_step
        self.valid_steps = 0

        self.self_attns = []  # store the all attns
        self.cross_attns = []

        self.self_attns_step = []  # store the attns in each step
        self.cross_attns_step = []

    def after_step(self):
        if self.cur_step > self.min_step and self.cur_step < self.max_step:
            self.valid_steps += 1
            if len(self.self_attns) == 0:
                self.self_attns = self.self_attns_step
                self.cross_attns = self.cross_attns_step
            else:
                for i in range(len(self.self_attns)):
                    self.self_attns[i] += self.self_attns_step[i]
                    self.cross_attns[i] += self.cross_attns_step[i]
        self.self_attns_step.clear()
        self.cross_attns_step.clear()

    def forward(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        if attn.shape[1] <= 64 ** 2:  # avoid OOM
            if is_cross:
                self.cross_attns_step.append(attn)
            else:
                self.self_attns_step.append(attn)
        return super().forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)


def regiter_attention_editor_diffusers(model, editor: AttentionBase, hook=False):
    """
    Register a attention editor to Diffuser Pipeline, refer from [Prompt-to-Prompt]
    """
    def ca_forward(self, place_in_unet):
        def forward(x, encoder_hidden_states=None, attention_mask=None, context=None, mask=None):
            """
            The attention is similar to the original implementation of LDM CrossAttention class
            except adding some modifications on the attention
            """
            if self.processor.__class__.__name__ == 'IPAttnProcessor2_0':
                hidden_states = x
                residual = hidden_states
                end_pos = encoder_hidden_states.shape[1] - self.processor.num_tokens
                encoder_hidden_states, ip_hidden_states = encoder_hidden_states[:, :end_pos, :], encoder_hidden_states[:, end_pos:, :]
                
            if encoder_hidden_states is not None:
                context = encoder_hidden_states
            if attention_mask is not None:
                mask = attention_mask

            to_out = self.to_out
            if isinstance(to_out, nn.modules.container.ModuleList):
                to_out = self.to_out[0]
            else:
                to_out = self.to_out

            h = self.heads
            is_cross = context is not None
            context = context if is_cross else x
            q = self.to_q(x)
            q = rearrange(q, 'b n (h d) -> (b h) n d', h=h)
            
            if editor.__class__.__name__ == 'AttentionBase' or hook or is_cross or editor.cur_step not in editor.step_idx or editor.cur_att_layer // 2 not in editor.layer_idx:
                k = self.to_k(context)
                v = self.to_v(context)
                k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (k, v))
                sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale
                if mask is not None:
                    mask = rearrange(mask, 'b ... -> b (...)')
                    max_neg_value = -torch.finfo(sim.dtype).max
                    mask = repeat(mask, 'b j -> (b h) () j', h=h)
                    mask = mask[:, None, :].repeat(h, 1, 1)
                    sim.masked_fill_(~mask, max_neg_value)
                attn = sim.softmax(dim=-1)
            elif editor.__class__.__name__ == 'MutualSelfAttentionControl':
                k = self.to_k(context)
                v = self.to_v(context)
                k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (k, v))
                sim = None
                attn = None
            elif editor.__class__.__name__ == 'MutualSelfAttentionControlCache':
                k = None
                v = None
                sim = None
                attn = None
                
            # the only difference
            out = editor(
                q, k, v, sim, attn, is_cross, place_in_unet,
                self.heads, scale=self.scale, hook=hook)
            
            if self.processor.__class__.__name__ == 'IPAttnProcessor2_0':
                # for ip-adapter
                ip_key = self.processor.to_k_ip(ip_hidden_states)
                ip_value = self.processor.to_v_ip(ip_hidden_states)
                
                # ip_key, ip_value = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (ip_key, ip_value))
                ip_key = self.head_to_batch_dim(ip_key)
                ip_value = self.head_to_batch_dim(ip_value)
                
                # ip_attention_probs = torch.einsum('b i d, b j d -> b i j', q, ip_key) * self.scale
                ip_attention_probs = self.get_attention_scores(q, ip_key, None)
                
                ip_hidden_states = torch.bmm(ip_attention_probs, ip_value)
                
                # ip_hidden_states = rearrange(ip_hidden_states, '(b h) n d -> b n (h d)', h=h)
                ip_hidden_states = self.batch_to_head_dim(ip_hidden_states)
                
                out = out + self.processor.scale * ip_hidden_states

                # linear proj
                out = self.to_out[0](out)
                # dropout
                out = self.to_out[1](out)
                # out = self.to_out(hidden_states)

                # if input_ndim == 4:
                #     hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

                if self.residual_connection:
                    out = out + residual
                out = out / self.rescale_output_factor
                
                return out
            else:
                return to_out(out)

        return forward

    def register_editor(net, count, place_in_unet):
        for name, subnet in net.named_children():
            if net.__class__.__name__ == 'Attention':  # spatial Transformer layer
                # if net.processor.__class__.__name__ == 'IPAttnProcessor2_0':
                #     return count + 1
                # print("change attn with ", net.processor.__class__.__name__)
                net.forward = ca_forward(net, place_in_unet)
                return count + 1
            elif hasattr(net, 'children'):
                count = register_editor(subnet, count, place_in_unet)
        return count

    cross_att_count = 0
    for net_name, net in model.unet.named_children():
        if "down" in net_name:
            cross_att_count += register_editor(net, 0, "down")
        elif "mid" in net_name:
            cross_att_count += register_editor(net, 0, "mid")
        elif "up" in net_name:
            cross_att_count += register_editor(net, 0, "up")
    editor.num_att_layers = cross_att_count


def regiter_attention_editor_ldm(model, editor: AttentionBase):
    """
    Register a attention editor to Stable Diffusion model, refer from [Prompt-to-Prompt]
    """
    def ca_forward(self, place_in_unet):
        def forward(x, encoder_hidden_states=None, attention_mask=None, context=None, mask=None):
            """
            The attention is similar to the original implementation of LDM CrossAttention class
            except adding some modifications on the attention
            """
            if encoder_hidden_states is not None:
                context = encoder_hidden_states
            if attention_mask is not None:
                mask = attention_mask

            to_out = self.to_out
            if isinstance(to_out, nn.modules.container.ModuleList):
                to_out = self.to_out[0]
            else:
                to_out = self.to_out

            h = self.heads
            q = self.to_q(x)
            is_cross = context is not None
            context = context if is_cross else x
            k = self.to_k(context)
            v = self.to_v(context)
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

            sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale

            if mask is not None:
                mask = rearrange(mask, 'b ... -> b (...)')
                max_neg_value = -torch.finfo(sim.dtype).max
                mask = repeat(mask, 'b j -> (b h) () j', h=h)
                mask = mask[:, None, :].repeat(h, 1, 1)
                sim.masked_fill_(~mask, max_neg_value)

            attn = sim.softmax(dim=-1)
            # the only difference
            out = editor(
                q, k, v, sim, attn, is_cross, place_in_unet,
                self.heads, scale=self.scale)

            return to_out(out)

        return forward

    def register_editor(net, count, place_in_unet):
        for name, subnet in net.named_children():
            if net.__class__.__name__ == 'CrossAttention':  # spatial Transformer layer
                net.forward = ca_forward(net, place_in_unet)
                return count + 1
            elif hasattr(net, 'children'):
                count = register_editor(subnet, count, place_in_unet)
        return count
    
    cross_att_count = 0
    for net_name, net in model.model.diffusion_model.named_children():
        if "input" in net_name:
            cross_att_count += register_editor(net, 0, "input")
        elif "middle" in net_name:
            cross_att_count += register_editor(net, 0, "middle")
        elif "output" in net_name:
            cross_att_count += register_editor(net, 0, "output")
    editor.num_att_layers = cross_att_count
