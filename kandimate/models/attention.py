# Adapted from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention.py
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from diffusers.models.attention_processor import  Attention
from diffusers.utils.import_utils import is_xformers_available


from .processors import XFormersAttnAddedKVProcessor, AttnAddedKVProcessor, AttnAddedKVProcessor2_0

if is_xformers_available():
    import xformers
    import xformers.ops
else:
    xformers = None

class InflatedAttention(Attention):
    def set_use_memory_efficient_attention_xformers(
        self, use_memory_efficient_attention_xformers, attention_op=None
    ):
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

            processor = XFormersAttnAddedKVProcessor(
                attention_op=attention_op,
            )
            self.set_processor(processor)
        else:
            processor = (
                AttnAddedKVProcessor2_0() if hasattr(F, "scaled_dot_product_attention") else AttnAddedKVProcessor()
            )
            self.set_processor(processor)

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, **cross_attention_kwargs):
        _, _, video_length, height, width = hidden_states.shape
        hidden_states = rearrange(hidden_states, "b c f h w -> (b f) (h w) c", f=video_length)
        
        if encoder_hidden_states is not None:
            encoder_hidden_states = repeat(encoder_hidden_states, "b n c -> (b f) n c", f=video_length)

        hidden_states = self.processor(
            self,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )
        hidden_states = rearrange(hidden_states, "(b f) (h w) c -> b c f h w", f=video_length, h=height, w=width).contiguous()

        return hidden_states