from dataclasses import dataclass
from typing import Optional, Callable

import torch
from torch import nn
import torch.nn.functional as F
from diffusers.utils.import_utils import is_xformers_available
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import BaseOutput
from diffusers.models.attention import FeedForward
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.attention_processor import Attention, AttnProcessor2_0

from einops import rearrange, repeat
import math

# from .processors import AttnAddedKVProcessor2_0, AttnAddedKVProcessor

if is_xformers_available():
    import xformers
    import xformers.ops
else:
    xformers = None

def zero_module(module):
    # Zero out the parameters of a module and return it.
    for p in module.parameters():
        p.detach().zero_()
    return module


dataclass
class TransformerTemporalModelOutput(BaseOutput):
    sample: torch.FloatTensor


def get_motion_module(
    in_channels,
    processor,
    motion_module_kwargs: dict
):
    return TransformerTemporalModel(in_channels=in_channels, processor=processor, **motion_module_kwargs)    


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout = 0.0, max_len = 128):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerTemporalModel(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        in_channels,
        processor,
        num_attention_heads: int = 8,
        attention_head_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        attention_bias: bool = False,
        activation_fn: str = "geglu",
        cross_attention_dim = None,
        temporal_position_encoding_max_len = 128,
        zero_initialize = True,
        use_pos_embed = False,
        *args, **kwargs
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        inner_dim = num_attention_heads * attention_head_dim

        self.in_channels = in_channels

        self.norm = torch.nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=1e-6, affine=True)
        self.proj_in = nn.Linear(in_channels, inner_dim)

        # 3. Define transformers blocks
        self.transformer_blocks = nn.ModuleList(
            [
                TemporalTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    cross_attention_dim=cross_attention_dim,
                    processor=processor,
                    norm_num_groups=norm_num_groups,
                    temporal_position_encoding=use_pos_embed,
                    temporal_position_encoding_max_len=temporal_position_encoding_max_len,
                )
                for _ in range(num_layers)
            ]
        )

        self.proj_out = nn.Linear(inner_dim, in_channels)

        if zero_initialize:
            self.proj_out = zero_module(self.proj_out)
    
    def forward(self, hidden_states, encoder_hidden_states=None, timestep=None, attention_mask=None):
        assert hidden_states.dim() == 5, f"Expected hidden_states to have ndim=5, but got ndim={hidden_states.dim()}."
        video_length = hidden_states.shape[2]
        hidden_states = rearrange(hidden_states, "b c f h w -> (b f) c h w")

        batch, channel, height, weight = hidden_states.shape
        residual = hidden_states

        hidden_states = self.norm(hidden_states)

        hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * weight, channel)
        hidden_states = self.proj_in(hidden_states)

        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, encoder_hidden_states=encoder_hidden_states, video_length=video_length)
        
        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states.reshape(batch, height, weight, channel).permute(0, 3, 1, 2).contiguous()

        output = hidden_states + residual
        output = rearrange(output, "(b f) c h w -> b c f h w", f=video_length)
        
        return output
    

class TemporalTransformerBlock(nn.Module):
    def __init__(
        self,
        query_dim,
        num_attention_heads,
        attention_head_dim,
        processor,
        num_layers=2,
        dropout = 0.0,
        norm_num_groups=32,
        activation_fn = "geglu",
        attention_bias = False,
        upcast_attention = False,
        cross_attention_dim = None,

        temporal_position_encoding = False,
        temporal_position_encoding_max_len = 24,
    ):
        super().__init__()

        attention_blocks = []
        norms = []
        
        for _ in range(num_layers):
            attention_blocks.append(
                VersatileAttention(
                    query_dim=query_dim,
                    heads=num_attention_heads,
                    dim_head=attention_head_dim,
                    dropout=dropout,
                    bias=attention_bias,
                    upcast_attention=upcast_attention,
                    cross_attention_dim=cross_attention_dim,
                    norm_num_groups=norm_num_groups,
                    processor=processor,

                    temporal_position_encoding=temporal_position_encoding,
                    temporal_position_encoding_max_len=temporal_position_encoding_max_len,
                )
            )
            norms.append(nn.LayerNorm(query_dim))
            
        self.attention_blocks = nn.ModuleList(attention_blocks)
        self.norms = nn.ModuleList(norms)

        self.ff = FeedForward(query_dim, dropout=dropout, activation_fn=activation_fn)
        self.ff_norm = nn.LayerNorm(query_dim)


    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, video_length=None):
        for attention_block, norm in zip(self.attention_blocks, self.norms):
            norm_hidden_states = norm(hidden_states)
            hidden_states = attention_block(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states if attention_block.is_cross_attention else None,
                video_length=video_length,
            ) + hidden_states
            
        hidden_states = self.ff(self.ff_norm(hidden_states)) + hidden_states
        
        output = hidden_states  
        return output



class VersatileAttention(Attention):
    def __init__(
            self,
            temporal_position_encoding = False,
            temporal_position_encoding_max_len = 24,            
            *args, **kwargs
        ):
        super().__init__(*args, **kwargs)

        self.is_cross_attention = kwargs["cross_attention_dim"] is not None
        
        self.pos_encoder = PositionalEncoding(
            kwargs["query_dim"],  #TODO ??
            dropout=0., 
            max_len=temporal_position_encoding_max_len
        ) if temporal_position_encoding else None

    def set_use_memory_efficient_attention_xformers(
        self, use_memory_efficient_attention_xformers: bool, attention_op: Optional[Callable] = None
    ):
        # print('-'*50, 'CUSTOM ATTENTION')
        if use_memory_efficient_attention_xformers:
            if not is_xformers_available():
                raise ModuleNotFoundError(
                    (
                        "Refer to https://github.com/facebookresearch/xformers for more information on how to install"
                        " xformers"
                    ),
                    name="xformers",
                )
            elif not torch.cuda.is_available():
                raise ValueError(
                    "torch.cuda.is_available() should be True but is False. xformers' memory efficient attention is"
                    " only available for GPU "
                )
            else:
                try:
                    # Make sure we can run the memory efficient attention
                    _ = xformers.ops.memory_efficient_attention(
                        torch.randn((1, 2, 40), device="cuda"),
                        torch.randn((1, 2, 40), device="cuda"),
                        torch.randn((1, 2, 40), device="cuda"),
                    )
                except Exception as e:
                    raise e

            # processor = XFormersAttnProcessor(
            #     attention_op=attention_op,
            # )
            # self.set_processor(processor)

            # processor = (
            #     AttnAddedKVProcessor2_0() if hasattr(F, "scaled_dot_product_attention") else AttnAddedKVProcessor()
            # )
            processor = AttnProcessor2_0()
            self.set_processor(processor)
        else:
            # processor = (
            #     AttnAddedKVProcessor2_0() if hasattr(F, "scaled_dot_product_attention") else AttnAddedKVProcessor()
            # )
            processor = AttnProcessor2_0()
            self.set_processor(processor)

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, video_length=None, **cross_attention_kwargs):
        d = hidden_states.shape[1]
        hidden_states = rearrange(hidden_states, "(b f) d c -> (b d) f c", f=video_length)  # batch * frames_len, height * weight, channel
        
        if self.pos_encoder is not None:
            hidden_states = self.pos_encoder(hidden_states)
        
        encoder_hidden_states = repeat(encoder_hidden_states, "b n c -> (b d) n c", d=d) if encoder_hidden_states is not None else encoder_hidden_states

        hidden_states = self.processor(
            self,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )
        hidden_states = rearrange(hidden_states, "(b d) f c -> (b f) d c", d=d)
        return hidden_states
