import argparse
import datetime
import os

import torch
from omegaconf import OmegaConf
from transformers import CLIPTextModelWithProjection, CLIPTokenizer
from diffusers import KandinskyV22PriorPipeline
from diffusers import DDIMScheduler, DDPMScheduler, VQModel

from kandimate.models.unet import UNet3DConditionModel
from kandimate.pipelines.pipeline_kandimation import KandimatePipeline
from kandimate.utils.util import save_videos_grid

from pathlib import Path


def main(args):
    time_str = datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
    savedir = f"samples/{Path(args.config).stem}-{time_str}"
    os.makedirs(savedir)
    config = OmegaConf.load(args.config)

    tokenizer = CLIPTokenizer.from_pretrained(
        config['kandinsky_prior'], 
        subfolder='tokenizer',
    )
    text_encoder = CLIPTextModelWithProjection.from_pretrained(
        config['kandinsky_prior'], 
        subfolder='text_encoder',
        torch_dtype=torch.float16,
    )
    scheduler = DDPMScheduler.from_pretrained(
        config['kandinsky_decoder'], 
        subfolder='scheduler'
    )
    movq = VQModel.from_pretrained(
        config['kandinsky_decoder'], 
        subfolder="movq", 
        torch_dtype=torch.float16,
    )
    unet = UNet3DConditionModel.from_pretrained_2d(
        config['kandinsky_decoder'], 
        subfolder="unet",
        unet_additional_kwargs=config['unet_additional_kwargs'],
    )
    unet = unet.to(dtype=torch.float16)

    pipeline = KandimatePipeline(
        movq=movq, 
        text_encoder=text_encoder, 
        tokenizer=tokenizer, 
        unet=unet,
        scheduler=scheduler,
    )

    motion_module_state_dict = torch.load(config['motion_module'], map_location='cpu')
    state_dict = {}
    for name, tensor in motion_module_state_dict['state_dict'].items():
        state_dict[name.replace('module.', '')] = tensor
    missing, unexpected = pipeline.unet.load_state_dict(state_dict, strict=False)
    assert len(unexpected) == 0

    pipeline.to('cuda')

    pipe_prior = KandinskyV22PriorPipeline.from_pretrained(
        config['kandinsky_prior'], torch_dtype=torch.float16
    ).to('cuda')

    prompts = config['prompts']

    n_prompts = config.get('n_prompts', '')
    n_prompts = [n_prompts] * len(prompts) if isinstance(n_prompts, str) else n_prompts
    config['n_prompts'] = n_prompts

    random_seeds = config.get('random_seeds', 17)
    random_seeds = [random_seeds] * len(prompts) if isinstance(random_seeds, int) else random_seeds
    config['random_seeds'] = random_seeds        
    
    samples = []
    for prompt, negative_prompt, random_seed in zip(prompts, n_prompts, random_seeds):
        
        generator = torch.Generator(device='cuda').manual_seed(random_seed)
        print(f'current seed: {generator.initial_seed()}')
        print(f'sampling "{prompt}" ...')
        image_emb, zero_image_emb = pipe_prior(
            prompt=prompt, negative_prompt=negative_prompt, generator=generator, 
        ).to_tuple()

        sample = pipeline(
            prompt,
            image_embeds = image_emb.to(dtype=torch.float16),
            negative_image_embeds = zero_image_emb.to(dtype=torch.float16),
            negative_prompt = negative_prompt,
            num_inference_steps = config['num_inference_steps'],
            guidance_scale = config['guidance_scale'],
            width = config['img_w'],
            height = config['img_h'],
            video_length = config['video_length'],
            generator = generator,
            use_progress_bar = True,
        ).videos

        samples.append(sample)

        prompt = "-".join((prompt.replace("/", "").split(" ")[:10]))
        save_videos_grid(sample, f"{savedir}/sample/{prompt}.gif")
        print(f"save to {savedir}/sample/{prompt}.gif")

    samples = torch.concat(samples)
    save_videos_grid(samples, f"{savedir}/sample.gif", n_rows=4)
    OmegaConf.save(config, f"{savedir}/config.yaml")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()  
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args)
