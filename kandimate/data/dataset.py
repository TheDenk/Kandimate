import os, random
import glob
import numpy as np
import pandas as pd
from decord import VideoReader

import torch
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from kandimate.utils.util import zero_rank_print



class WebVid10M(Dataset):
    def __init__(
            self,
            csv_path, video_folder,
            sample_size=256, sample_stride=4, sample_n_frames=16,
            is_image=False,
        ):

        self.video_folder = video_folder
        self.sample_stride = sample_stride
        self.sample_n_frames = sample_n_frames
        self.is_image = is_image
        
        self.videos_paths = glob.glob(f'{video_folder}/*.mp4')
        self.length = len(self.videos_paths)
        
        df = pd.read_csv(csv_path)
        df = df.set_index('videoid')
        
        video_names = [int(os.path.basename(x).split('.')[0]) for x in self.videos_paths]
        self.df = df[df.index.isin(video_names)]
        
        zero_rank_print(f"data scale: {self.length}")
        
        sample_size = tuple(sample_size) if not isinstance(sample_size, int) else (sample_size, sample_size)
        self.pixel_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize(sample_size[0], antialias=False),
            transforms.CenterCrop(sample_size),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])
    
    def get_batch(self, idx):
        video_path = self.videos_paths[idx]
        video_id = int(os.path.basename(video_path).split('.')[0])
        name = self.df.loc[video_id]['name']
        video_reader = VideoReader(video_path)
        video_length = len(video_reader)
        
        if not self.is_image:
            clip_length = min(video_length, (self.sample_n_frames - 1) * self.sample_stride + 1)
            start_idx   = random.randint(0, video_length - clip_length)
            batch_index = np.linspace(start_idx, start_idx + clip_length - 1, self.sample_n_frames, dtype=int)
        else:
            batch_index = [random.randint(0, video_length - 1)]

        pixel_values = torch.from_numpy(video_reader.get_batch(batch_index).asnumpy()).permute(0, 3, 1, 2).contiguous()
        pixel_values = pixel_values / 255.
        del video_reader

        if self.is_image:
            pixel_values = pixel_values[0]
         # print(f'Collect {pixel_values.shape} | {name}')
        return pixel_values, name

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        while True:
            try:
                pixel_values, name = self.get_batch(idx)
                break

            except Exception as e:
                zero_rank_print(e)
                idx = random.randint(0, self.length-1)

        pixel_values = self.pixel_transforms(pixel_values)
        sample = dict(pixel_values=pixel_values, text=name)
        return sample


class WebVid10MLowMem(Dataset):
    def __init__(
            self,
            csv_path, video_folder, embeds_path,
            sample_size=256, sample_stride=4, sample_n_frames=16, is_image=False,
        ):

        self.video_folder = video_folder
        self.embeds_path = embeds_path
        self.sample_stride = sample_stride
        self.sample_n_frames = sample_n_frames
        self.is_image = is_image
        
        self.videos_paths = glob.glob(f'{video_folder}/*.mp4')
        self.length = len(self.videos_paths)
        
        df = pd.read_csv(csv_path)
        df = df.set_index('videoid')
        
        video_names = [int(os.path.basename(x).split('.')[0]) for x in self.videos_paths]
        self.df = df[df.index.isin(video_names)]

        self.embeds = torch.load(embeds_path)
        
        zero_rank_print(f"data scale: {self.length}")
        
        sample_size = tuple(sample_size) if not isinstance(sample_size, int) else (sample_size, sample_size)
        self.pixel_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize(sample_size[0], antialias=False),
            transforms.RandomCrop(sample_size),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])
    
    def get_batch(self, idx):
        video_path = self.videos_paths[idx]
        video_id = int(os.path.basename(video_path).split('.')[0])
        video_reader = VideoReader(video_path)
        video_length = len(video_reader)
        
        if not self.is_image:
            clip_length = min(video_length, (self.sample_n_frames - 1) * self.sample_stride + 1)
            start_idx   = random.randint(0, video_length - clip_length)
            batch_index = np.linspace(start_idx, start_idx + clip_length - 1, self.sample_n_frames, dtype=int)
        else:
            batch_index = [random.randint(0, video_length - 1)]

        pixel_values = torch.from_numpy(video_reader.get_batch(batch_index).asnumpy()).permute(0, 3, 1, 2).contiguous()
        pixel_values = pixel_values / 255.
        del video_reader

        if self.is_image:
            pixel_values = pixel_values[0]
        
        image_embeds = self.embeds[video_id]["image_embeds"]
        prompt_embeds = self.embeds[video_id]["prompt_embeds"]
        
        return pixel_values, image_embeds, prompt_embeds

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        while True:
            try:
                pixel_values, image_embeds, prompt_embeds = self.get_batch(idx)
                break

            except Exception as e:
                zero_rank_print(e)
                idx = random.randint(0, self.length-1)

        pixel_values = self.pixel_transforms(pixel_values)
        sample = dict(
            pixel_values=pixel_values, 
            image_embeds=image_embeds, 
            prompt_embeds=prompt_embeds,
            zero_image_embeds = self.embeds["zero_image_embeds"],
            zero_prompt_embeds = self.embeds["zero_prompt_embeds"],
        )
        return sample
