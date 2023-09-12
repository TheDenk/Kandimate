# Adapted from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/unet_2d_blocks.py
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from diffusers.utils import is_torch_version
from diffusers.models.attention_processor import AttnProcessor2_0
from .attention import InflatedAttention
from .resnet import ResnetBlock3D
from .motion_module import get_motion_module
from .processors import AttnAddedKVProcessor2_0, AttnAddedKVProcessor

import pdb


def get_down_block(
    down_block_type,
    num_layers,
    in_channels,
    out_channels,
    temb_channels,
    add_downsample,
    resnet_eps,
    resnet_act_fn,
    resnet_groups=None,
    cross_attention_dim=None,
    only_cross_attention=False,
    resnet_time_scale_shift="default",
    resnet_out_scale_factor=1.0,
    cross_attention_norm=None,
    downsample_padding=1,
    attention_head_dim=1,
    dropout=0.0,

    use_motion_module=False,
    motion_module_kwargs={},
):

    down_block_type = down_block_type[7:] if down_block_type.startswith("UNetRes") else down_block_type
    if down_block_type == "ResnetDownsampleBlock3D":
        return ResnetDownsampleBlock3D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            dropout=dropout,
            add_downsample=add_downsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            resnet_time_scale_shift=resnet_time_scale_shift,
            output_scale_factor=resnet_out_scale_factor,

            use_motion_module=use_motion_module,
            motion_module_kwargs=motion_module_kwargs,
        )
    elif down_block_type == "SimpleCrossAttnDownBlock3D":
        if cross_attention_dim is None:
            raise ValueError("cross_attention_dim must be specified for SimpleCrossAttnDownBlock2D")
        return SimpleCrossAttnDownBlock3D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            dropout=dropout,
            add_downsample=add_downsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            cross_attention_dim=cross_attention_dim,
            attention_head_dim=attention_head_dim,
            resnet_time_scale_shift=resnet_time_scale_shift,
            output_scale_factor=resnet_out_scale_factor,
            only_cross_attention=only_cross_attention,
            cross_attention_norm=cross_attention_norm,

            use_motion_module=use_motion_module,
            motion_module_kwargs=motion_module_kwargs,
        )
    raise ValueError(f"{down_block_type} does not exist.")


def get_up_block(
    up_block_type,
    num_layers,
    in_channels,
    out_channels,
    prev_output_channel,
    temb_channels,
    add_upsample,
    resnet_eps,
    resnet_act_fn,
    resnet_groups=None,
    cross_attention_dim=None,
    only_cross_attention=False,
    resnet_time_scale_shift="default",
    resnet_out_scale_factor=1.0,
    cross_attention_norm=None,
    attention_head_dim=1,
    dropout=0.0,

    use_motion_module=False,
    motion_module_kwargs={},
):

    up_block_type = up_block_type[7:] if up_block_type.startswith("UNetRes") else up_block_type
    if up_block_type == "ResnetUpsampleBlock3D":
        return ResnetUpsampleBlock3D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            prev_output_channel=prev_output_channel,
            temb_channels=temb_channels,
            dropout=dropout,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            resnet_time_scale_shift=resnet_time_scale_shift,
            output_scale_factor=resnet_out_scale_factor,
            
            use_motion_module=use_motion_module,
            motion_module_kwargs=motion_module_kwargs,
        )
    elif up_block_type == "SimpleCrossAttnUpBlock3D":
        if cross_attention_dim is None:
            raise ValueError("cross_attention_dim must be specified for SimpleCrossAttnUpBlock2D")
        return SimpleCrossAttnUpBlock3D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            prev_output_channel=prev_output_channel,
            temb_channels=temb_channels,
            dropout=dropout,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            cross_attention_dim=cross_attention_dim,
            attention_head_dim=attention_head_dim,
            resnet_time_scale_shift=resnet_time_scale_shift,
            output_scale_factor=resnet_out_scale_factor,
            only_cross_attention=only_cross_attention,
            cross_attention_norm=cross_attention_norm,

            use_motion_module=use_motion_module,
            motion_module_kwargs=motion_module_kwargs,
        )
    raise ValueError(f"{up_block_type} does not exist.")

class ResnetDownsampleBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor=1.0,
        add_downsample=True,

        use_motion_module=None,
        motion_module_kwargs=None,
    ):
        super().__init__()
        resnets = []
        motion_modules = []

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock3D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
            processor = AttnProcessor2_0()
            motion_modules.append(
                get_motion_module(
                    in_channels=out_channels,
                    processor=processor,
                    motion_module_kwargs=motion_module_kwargs,
                ) if use_motion_module else None
            )

        self.resnets = nn.ModuleList(resnets)
        self.motion_modules = nn.ModuleList(motion_modules)

        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    ResnetBlock3D(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        temb_channels=temb_channels,
                        eps=resnet_eps,
                        groups=resnet_groups,
                        dropout=dropout,
                        time_embedding_norm=resnet_time_scale_shift,
                        non_linearity=resnet_act_fn,
                        output_scale_factor=output_scale_factor,
                        pre_norm=resnet_pre_norm,
                        down=True,
                    )
                ]
            )
        else:
            self.downsamplers = None

        self.gradient_checkpointing = False

    def forward(self, hidden_states, temb=None, scale: float = 1.0):
        output_states = ()

        for resnet, motion_module in zip(self.resnets, self.motion_modules):
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                if is_torch_version(">=", "1.11.0"):
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(resnet), hidden_states, temb, use_reentrant=False
                    )
                else:
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(resnet), hidden_states, temb
                    )
                if motion_module is not None:
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(motion_module), hidden_states.requires_grad_(), None, temb
                    )
                    
            else:
                hidden_states = resnet(hidden_states, temb, scale)
                hidden_states = motion_module(
                    hidden_states, 
                    encoder_hidden_states=None, 
                    timestep=temb
                ) if motion_module is not None else hidden_states

            output_states = output_states + (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states, temb, scale)

            output_states = output_states + (hidden_states,)

        return hidden_states, output_states


class SimpleCrossAttnDownBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        attention_head_dim=1,
        cross_attention_dim=1280,
        output_scale_factor=1.0,
        add_downsample=True,
        only_cross_attention=False,
        cross_attention_norm=None,
        
        use_motion_module=None,
        motion_module_kwargs=None,
    ):
        super().__init__()

        self.has_cross_attention = True

        resnets = []
        attentions = []
        motion_modules = []

        self.attention_head_dim = attention_head_dim
        self.num_heads = out_channels // self.attention_head_dim

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock3D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )

            processor = (
                AttnAddedKVProcessor2_0() if hasattr(F, "scaled_dot_product_attention") else AttnAddedKVProcessor()
            )
            # processor = AttnAddedKVProcessor()

            attentions.append(
                InflatedAttention(
                    query_dim=out_channels,
                    cross_attention_dim=out_channels,
                    heads=self.num_heads,
                    dim_head=attention_head_dim,
                    added_kv_proj_dim=cross_attention_dim,
                    norm_num_groups=resnet_groups,
                    bias=True,
                    upcast_softmax=True,
                    only_cross_attention=only_cross_attention,
                    cross_attention_norm=cross_attention_norm,
                    processor=processor
                )
            )
            processor = AttnProcessor2_0()
            motion_modules.append(
                get_motion_module(
                    in_channels=out_channels,
                    processor=processor,
                    motion_module_kwargs=motion_module_kwargs,
                ) if use_motion_module else None
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)
        self.motion_modules = nn.ModuleList(motion_modules)

        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    ResnetBlock3D(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        temb_channels=temb_channels,
                        eps=resnet_eps,
                        groups=resnet_groups,
                        dropout=dropout,
                        time_embedding_norm=resnet_time_scale_shift,
                        non_linearity=resnet_act_fn,
                        output_scale_factor=output_scale_factor,
                        pre_norm=resnet_pre_norm,
                        down=True,
                    )
                ]
            )
        else:
            self.downsamplers = None

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        temb: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
    ):
        output_states = ()
        cross_attention_kwargs = cross_attention_kwargs if cross_attention_kwargs is not None else {}

        lora_scale = cross_attention_kwargs.get("scale", 1.0)

        if attention_mask is None:
            # if encoder_hidden_states is defined: we are doing cross-attn, so we should use cross-attn mask.
            mask = None if encoder_hidden_states is None else encoder_attention_mask
        else:
            # when attention_mask is defined: we don't even check for encoder_attention_mask.
            # this is to maintain compatibility with UnCLIP, which uses 'attention_mask' param for cross-attn masks.
            # TODO: UnCLIP should express cross-attn mask via encoder_attention_mask param instead of via attention_mask.
            #       then we can simplify this whole if/else block to:
            #         mask = attention_mask if encoder_hidden_states is None else encoder_attention_mask
            mask = attention_mask

        for resnet, attn, motion_module in zip(self.resnets, self.attentions, self.motion_modules):
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(resnet), hidden_states, temb)
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=mask,
                    **cross_attention_kwargs,
                )
                if motion_module is not None:
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(motion_module), 
                        hidden_states.requires_grad_(), 
                        encoder_hidden_states,
                        temb
                    )
            else:
                hidden_states = resnet(hidden_states, temb, scale=lora_scale)

                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=mask,
                    **cross_attention_kwargs,
                )
                hidden_states = motion_module(
                    hidden_states, 
                    encoder_hidden_states=encoder_hidden_states, 
                    timestep=temb
                ) if motion_module is not None else hidden_states

            output_states = output_states + (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states, temb, scale=lora_scale)

            output_states = output_states + (hidden_states,)

        return hidden_states, output_states


class SimpleCrossAttnUpBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        prev_output_channel: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        attention_head_dim=1,
        cross_attention_dim=1280,
        output_scale_factor=1.0,
        add_upsample=True,
        only_cross_attention=False,
        cross_attention_norm=None,

        use_motion_module=None,
        motion_module_kwargs=None,
    ):
        super().__init__()
        resnets = []
        attentions = []
        motion_modules = []

        self.has_cross_attention = True
        self.attention_head_dim = attention_head_dim

        self.num_heads = out_channels // self.attention_head_dim

        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                ResnetBlock3D(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
            processor = (
                AttnAddedKVProcessor2_0() if hasattr(F, "scaled_dot_product_attention") else AttnAddedKVProcessor()
            )
            # processor = AttnAddedKVProcessor()
            attentions.append(
                InflatedAttention(
                    query_dim=out_channels,
                    cross_attention_dim=out_channels,
                    heads=self.num_heads,
                    dim_head=self.attention_head_dim,
                    added_kv_proj_dim=cross_attention_dim,
                    norm_num_groups=resnet_groups,
                    bias=True,
                    upcast_softmax=True,
                    only_cross_attention=only_cross_attention,
                    cross_attention_norm=cross_attention_norm,
                    processor=processor
                )
            )
            processor = AttnProcessor2_0()
            motion_modules.append(
                get_motion_module(
                    in_channels=out_channels,
                    processor=processor,
                    motion_module_kwargs=motion_module_kwargs,
                ) if use_motion_module else None
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)
        self.motion_modules = nn.ModuleList(motion_modules)

        if add_upsample:
            self.upsamplers = nn.ModuleList(
                [
                    ResnetBlock3D(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        temb_channels=temb_channels,
                        eps=resnet_eps,
                        groups=resnet_groups,
                        dropout=dropout,
                        time_embedding_norm=resnet_time_scale_shift,
                        non_linearity=resnet_act_fn,
                        output_scale_factor=output_scale_factor,
                        pre_norm=resnet_pre_norm,
                        up=True,
                    )
                ]
            )
        else:
            self.upsamplers = None

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        res_hidden_states_tuple: Tuple[torch.FloatTensor, ...],
        temb: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        upsample_size: Optional[int] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
    ):
        cross_attention_kwargs = cross_attention_kwargs if cross_attention_kwargs is not None else {}

        lora_scale = cross_attention_kwargs.get("scale", 1.0)
        if attention_mask is None:
            # if encoder_hidden_states is defined: we are doing cross-attn, so we should use cross-attn mask.
            mask = None if encoder_hidden_states is None else encoder_attention_mask
        else:
            # when attention_mask is defined: we don't even check for encoder_attention_mask.
            # this is to maintain compatibility with UnCLIP, which uses 'attention_mask' param for cross-attn masks.
            # TODO: UnCLIP should express cross-attn mask via encoder_attention_mask param instead of via attention_mask.
            #       then we can simplify this whole if/else block to:
            #         mask = attention_mask if encoder_hidden_states is None else encoder_attention_mask
            mask = attention_mask

        for resnet, attn, motion_module in zip(self.resnets, self.attentions, self.motion_modules):
            # resnet
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(resnet), hidden_states, temb)
                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=mask,
                    **cross_attention_kwargs,
                )
                if motion_module is not None:
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(motion_module), 
                        hidden_states.requires_grad_(), 
                        encoder_hidden_states,
                        temb
                    )
            else:
                hidden_states = resnet(hidden_states, temb, scale=lora_scale)

                hidden_states = attn(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=mask,
                    **cross_attention_kwargs,
                )
                hidden_states = motion_module(
                    hidden_states, 
                    encoder_hidden_states=encoder_hidden_states, 
                    timestep=temb
                ) if motion_module is not None else hidden_states

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, temb, scale=lora_scale)

        return hidden_states


class ResnetUpsampleBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        prev_output_channel: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor=1.0,
        add_upsample=True,

        use_motion_module=None,
        motion_module_kwargs=None,
    ):
        super().__init__()
        resnets = []
        motion_modules = []

        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                ResnetBlock3D(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
            processor = AttnProcessor2_0()
            motion_modules.append(
                get_motion_module(
                    in_channels=out_channels,
                    processor=processor,
                    motion_module_kwargs=motion_module_kwargs,
                ) if use_motion_module else None
            )

        self.resnets = nn.ModuleList(resnets)
        self.motion_modules = nn.ModuleList(motion_modules)

        if add_upsample:
            self.upsamplers = nn.ModuleList(
                [
                    ResnetBlock3D(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        temb_channels=temb_channels,
                        eps=resnet_eps,
                        groups=resnet_groups,
                        dropout=dropout,
                        time_embedding_norm=resnet_time_scale_shift,
                        non_linearity=resnet_act_fn,
                        output_scale_factor=output_scale_factor,
                        pre_norm=resnet_pre_norm,
                        up=True,
                    )
                ]
            )
        else:
            self.upsamplers = None

        self.gradient_checkpointing = False

    def forward(self, hidden_states, res_hidden_states_tuple, temb=None, upsample_size=None, scale: float = 1.0):
        for resnet, motion_module in zip(self.resnets, self.motion_modules):
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)

            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                if is_torch_version(">=", "1.11.0"):
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(resnet), hidden_states, temb, use_reentrant=False
                    )
                else:
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(resnet), hidden_states, temb
                    )
                if motion_module is not None:
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(motion_module), hidden_states.requires_grad_(), None, temb
                    )
            else:
                hidden_states = resnet(hidden_states, temb, scale=scale)
                hidden_states = motion_module(
                    hidden_states, 
                    encoder_hidden_states=None, 
                    timestep=temb
                ) if motion_module is not None else hidden_states

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, temb, scale=scale)

        return hidden_states


class UNetMidBlock3DSimpleCrossAttn(nn.Module):
    def __init__(
        self,
        in_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        attention_head_dim=1,
        output_scale_factor=1.0,
        cross_attention_dim=1280,
        only_cross_attention=False,
        cross_attention_norm=None,

        use_motion_module=None,
        motion_module_kwargs=None,
    ):
        super().__init__()

        self.has_cross_attention = True

        self.attention_head_dim = attention_head_dim
        resnet_groups = resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)

        self.num_heads = in_channels // self.attention_head_dim

        # there is always at least one resnet
        resnets = [
            ResnetBlock3D(
                in_channels=in_channels,
                out_channels=in_channels,
                temb_channels=temb_channels,
                eps=resnet_eps,
                groups=resnet_groups,
                dropout=dropout,
                time_embedding_norm=resnet_time_scale_shift,
                non_linearity=resnet_act_fn,
                output_scale_factor=output_scale_factor,
                pre_norm=resnet_pre_norm,
            )
        ]
        attentions = []
        motion_modules = []

        for _ in range(num_layers):
            processor = (
                AttnAddedKVProcessor2_0() if hasattr(F, "scaled_dot_product_attention") else AttnAddedKVProcessor()
            )
            # processor = AttnAddedKVProcessor()
            attentions.append(
                InflatedAttention(
                    query_dim=in_channels,
                    cross_attention_dim=in_channels,
                    heads=self.num_heads,
                    dim_head=self.attention_head_dim,
                    added_kv_proj_dim=cross_attention_dim,
                    norm_num_groups=resnet_groups,
                    bias=True,
                    upcast_softmax=True,
                    only_cross_attention=only_cross_attention,
                    cross_attention_norm=cross_attention_norm,
                    processor=processor
                )
            )
            processor = AttnProcessor2_0()
            motion_modules.append(
                get_motion_module(
                    in_channels=in_channels,
                    processor=processor,
                    motion_module_kwargs=motion_module_kwargs,
                ) if use_motion_module else None
            )
            resnets.append(
                ResnetBlock3D(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)
        self.motion_modules = nn.ModuleList(motion_modules)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        temb: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
    ):
        cross_attention_kwargs = cross_attention_kwargs if cross_attention_kwargs is not None else {}
        lora_scale = cross_attention_kwargs.get("scale", 1.0)

        if attention_mask is None:
            mask = None if encoder_hidden_states is None else encoder_attention_mask
        else:
            mask = attention_mask

        hidden_states = self.resnets[0](hidden_states, temb, scale=lora_scale)
        for attn, resnet, motion_module in zip(self.attentions, self.resnets[1:], self.motion_modules):
            # attn
            hidden_states = attn(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=mask,
                **cross_attention_kwargs,
            )
            hidden_states = motion_module(hidden_states, temb, encoder_hidden_states=encoder_hidden_states) if motion_module is not None else hidden_states
            # resnet
            hidden_states = resnet(hidden_states, temb, scale=lora_scale)

        return hidden_states
