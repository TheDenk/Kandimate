import argparse
import os

import torch
import cv2
import imageio
import numpy as np
from omegaconf import OmegaConf


def pil_list_to_torch(images):
    tensor = [torch.from_numpy(x) for x in images]
    tensor = [x.permute(2, 0, 1).unsqueeze(0) for x in tensor]
    tensor = [x.float() / 255.0 for x in tensor]
    tensor = torch.cat(tensor)
    return tensor

def torch_to_np(tensor):
    images = tensor.permute(0, 2, 3, 1).numpy()
    images = np.clip(images * 255, 0, 255)
    images = images.astype(np.uint8)
    return images

def resize_gif(in_path, out_path, img_h=512, img_w=512, duration=60):
    gif_file = imageio.get_reader(in_path)
    images = [cv2.resize(x, (img_w, img_h)) for x in gif_file]
    imageio.mimsave(out_path, images, duration=duration)

def main(args):
    config = OmegaConf.load(args.config)
    os.makedirs(config['out_folder'], exist_ok=True)

    device = torch.device(config['device'])
    precision = torch.float32 if config['precision'] == 'fp32' else torch.float16
    interpolation_frames = config['interpolation_frames']

    model = torch.jit.load(config['model_path'], map_location='cpu')
    model.eval().to(device=device, dtype=precision)

    gif_file = imageio.get_reader(config['gif_path'])
    images = [np.array(x) for x in gif_file]
    tensor = pil_list_to_torch(images)

    prev_tensor = tensor[0:-1:1].to(precision).to(device)
    next_tensor = tensor[1::1].to(precision).to(device)
    dt = prev_tensor.new_full((1, 1), 0.5)

    with torch.no_grad():
        dt_images_mid = model(prev_tensor, next_tensor, dt).cpu()
        generated_mid = torch_to_np(dt_images_mid)
        
    if interpolation_frames > 1:
        with torch.no_grad():
            dt_images_prev = model(prev_tensor.to(device), dt_images_mid.to(device), dt).cpu()
            dt_images_next = model(dt_images_mid.to(device), next_tensor.to(device), dt).cpu()
            
        generated_prev = torch_to_np(dt_images_prev)
        generated_next = torch_to_np(dt_images_next)

    extended_images = [images[0]]

    if interpolation_frames == 1:
        for mid_img, orig_img in zip(generated_mid, images[1:]):
            extended_images.extend([mid_img, orig_img])
    elif interpolation_frames == 2:
        for prev_img, next_img, orig_img in zip(generated_prev, generated_next, images[1:]):
            extended_images.extend([prev_img, next_img, orig_img])
    elif interpolation_frames == 3:
        for prev_img, mid_img, next_img, orig_img in zip(generated_prev, generated_mid, generated_next, images[1:]):
            extended_images.extend([prev_img, mid_img, next_img, orig_img])

    
    out_h=config['out_h']
    out_w=config['out_w']

    out_folder = config['out_folder']
    out_name = config['out_filename'].split('.')[0]
    
    imageio.mimsave(f'{out_folder}/{out_name}_original_size.gif', extended_images, duration=60)

    for duration in config['durations']:
        resize_gif(
            in_path=f'{out_folder}/{out_name}_original_size.gif', 
            out_path=f'{out_folder}/{out_name}_{duration}.gif',
            img_h=out_h, 
            img_w=out_w,
            duration=duration,
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()  
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args)
