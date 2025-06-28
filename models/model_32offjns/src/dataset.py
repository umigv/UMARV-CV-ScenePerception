import sys
import numpy as np
import cv2
import sys
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import glob
import random
import os

class Dataset_Class(Dataset):

  # def __init__(self, device, split, datasets, transform, data_augmentation, data_augmentation_weight):
  def __init__(self, data, lookback, augment=False, device="cpu", label_input_threshold=.1):
    
    self.data = data
    self.lookback = lookback
    self.augment = augment
    self.device = device
    self.input_threshold = label_input_threshold
    self.data_size = 256
    self.dataset_dir = f"{os.getenv('ROOT_DIR')}/datasets"

    #Initialize default transforms
    self.default_data_transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Resize((self.data_size, self.data_size), antialias=None)
    ])
    self.default_label_transform = transforms.Compose([
      transforms.Resize((self.data_size, self.data_size), antialias=None),
    ])

    # Augmentation helpers
    random_color_jitter = transforms.ColorJitter(brightness=.2, contrast=.2, saturation=.15, hue=.1)
    random_rotation = transforms.RandomRotation(degrees=(-4,4))
    random_resized_crop = transforms.RandomResizedCrop(size=self.data_size, scale=(.5, .51), antialias=True)
    
    # Augmentation
    self.augmentation = [
      {
        'chance' : .5,
        'data' : transforms.RandomHorizontalFlip(p=1),
        'data_raw' : transforms.RandomHorizontalFlip(p=1),
        'label' : transforms.Compose([
          transforms.Lambda(lambda x: x.permute(2, 0, 1)),
          transforms.RandomHorizontalFlip(p=1),
          transforms.Lambda(lambda x: x.permute(1, 2, 0))
        ])
      },
      {
        'chance' : .75,
        'data' : random_color_jitter,
        'data_raw' : random_color_jitter,
        'label' : "None"
      },
      # {
      #   'chance' : .25,
      #   'data' : random_rotation,
      #   'data_raw' : random_rotation,
      #   'label' : transforms.Compose([
      #     transforms.Lambda(lambda x: x.permute(2, 0, 1)),
      #     random_rotation,
      #     transforms.Lambda(lambda x: x.permute(1, 2, 0))
      #   ])
      # },
      # {
      #   'chance' : .99,
      #   'data' : transforms.Compose([
      #     random_resized_crop,
      #     transforms.Resize((self.data_size, self.data_size), antialias=None)
      #   ]),
      #   'data_raw' : transforms.Compose([
      #     random_resized_crop,
      #     transforms.Resize((self.data_size, self.data_size), antialias=None)
      #   ]),
      #   'label' : transforms.Compose([
      #     transforms.Lambda(lambda x: x.permute(2, 0, 1)),                        
      #     random_resized_crop,
      #     transforms.Resize((self.data_size, self.data_size), antialias=None),
      #     transforms.Lambda(lambda x: x.permute(1, 2, 0))
      #   ])
      # }
    ]

  def __len__(self):
      return len(self.data)

  def __getitem__(self, idx):

    data_idx = self.data[idx]['idx']
    data_idx_str = str(data_idx).zfill(6)

    # Get data
    frames = []
    for i in range(0, self.lookback['count'] + 1):
      frame_data_idx = data_idx - (i * self.lookback['stride'])
      if frame_data_idx < 0:
        frame = frames[-1]
      else:
        frame_idx_str = str(frame_data_idx).zfill(6)
        frame_data_dir = f"{self.dataset_dir}/{self.data[idx]['dataset']}/data/{frame_idx_str}.jpg"
        frame = cv2.imread(frame_data_dir, cv2.IMREAD_COLOR)
        frame = self.default_data_transform(frame)
        frame = frame.to(self.device)
      frames.append(frame)

    data = torch.cat(frames, dim=0)
    data_raw = frames[0].detach().clone()

    #Get label
    label_background_dir = f"{self.dataset_dir}/{self.data[idx]['dataset']}/label/background/{data_idx_str}.jpg"
    label_lane_lines_dir = f"{self.dataset_dir}/{self.data[idx]['dataset']}/label/lane_lines/{data_idx_str}.jpg"
    label_drivable_area_dir = f"{self.dataset_dir}/{self.data[idx]['dataset']}/label/drivable_area/{data_idx_str}.jpg"
    label_cones_dir = f"{self.dataset_dir}/{self.data[idx]['dataset']}/label/cones/{data_idx_str}.jpg"

    label_background = torch.tensor(cv2.imread(label_background_dir, cv2.IMREAD_GRAYSCALE), device=self.device, dtype=torch.float32)
    label_lane_lines = torch.tensor(cv2.imread(label_lane_lines_dir, cv2.IMREAD_GRAYSCALE), device=self.device, dtype=torch.float32)
    label_drivable_area = torch.tensor(cv2.imread(label_drivable_area_dir, cv2.IMREAD_GRAYSCALE), device=self.device, dtype=torch.float32)
    label_cones = torch.tensor(cv2.imread(label_cones_dir, cv2.IMREAD_GRAYSCALE), device=self.device, dtype=torch.float32)
    
    label_shape = label_background.shape
    label = torch.zeros((label_shape[0], label_shape[1], 4), device=self.device, dtype=torch.long)

    label[..., 0] = torch.where(label_background > self.input_threshold, 1, 0)
    label[..., 1] = torch.where(label_lane_lines > self.input_threshold, 1, 0)
    label[..., 2] = torch.where(label_drivable_area > self.input_threshold, 1, 0)
    label[..., 3] = torch.where(label_cones > self.input_threshold, 1, 0)

    label = self.default_label_transform(label.permute(2, 0, 1)).permute(1, 2, 0)

    if self.augment:
      for augmentation in self.augmentation:
        if augmentation['chance'] > random.random():
          data = augmentation['data'](data)
          if augmentation['data_raw'] != "None":
            data_raw = augmentation['data_raw'](data_raw)
          if augmentation['label'] != "None":
            label = augmentation['label'](label)

    return data_raw, data, label