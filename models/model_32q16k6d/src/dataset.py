import cv2
import torch
import torchvision.transforms.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import random
import os

class Dataset_Class(Dataset):

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

    # Augmentation - Horizontal Flip
    chance = .5
    if self.augment and chance > random.random():
      augmentation_data = transforms.RandomHorizontalFlip(p=1)
      augmentation_data_raw = transforms.RandomHorizontalFlip(p=1)
      augmentation_label = transforms.Compose([
        transforms.Lambda(lambda x: x.permute(2, 0, 1)),
        transforms.RandomHorizontalFlip(p=1),
        transforms.Lambda(lambda x: x.permute(1, 2, 0))
      ])
      data = augmentation_data(data)
      data_raw = augmentation_data_raw(data_raw)
      label = augmentation_label(label)

    # Augmentation - Color Jitter
    chance = .66
    if self.augment and chance > random.random():
      jitter_variability = {
        "brightness" : .2,
        "contrast" : .2,
        "saturation" : .15,
        "hue" : .1
      }
      jitter = {
        "brightness" : random.uniform(1-jitter_variability['brightness'], 1+jitter_variability['brightness']),
        "contrast" : random.uniform(1-jitter_variability['contrast'], 1+jitter_variability['contrast']),
        "saturation" : random.uniform(1-jitter_variability['saturation'], 1+jitter_variability['saturation']),
        "hue" : random.uniform(-jitter_variability['hue'], jitter_variability['hue'])
      }
      augmentation_data = transforms.Compose([
        transforms.Lambda(lambda x: F.adjust_brightness(x, jitter['brightness'])),
        transforms.Lambda(lambda x: F.adjust_contrast(x, jitter['contrast'])),
        transforms.Lambda(lambda x: F.adjust_saturation(x, jitter['saturation'])),
        transforms.Lambda(lambda x: F.adjust_hue(x, jitter['hue'])),
      ])
      augmentation_data_raw = transforms.Compose([
        transforms.Lambda(lambda x: F.adjust_brightness(x, jitter['brightness'])),
        transforms.Lambda(lambda x: F.adjust_contrast(x, jitter['contrast'])),
        transforms.Lambda(lambda x: F.adjust_saturation(x, jitter['saturation'])),
        transforms.Lambda(lambda x: F.adjust_hue(x, jitter['hue'])),
      ])
      data = augmentation_data(data)
      data_raw = augmentation_data_raw(data_raw)

    # Augmentation - Random Rotation
    chance = .33
    if self.augment and chance > random.random():
      angle_variability = 7
      angle = random.uniform(-angle_variability, angle_variability)
      augmentation_data = transforms.Lambda(lambda x: F.rotate(x, angle))
      augmentation_data_raw = transforms.Lambda(lambda x: F.rotate(x, angle))
      augmentation_label = transforms.Compose([
        transforms.Lambda(lambda x: x.permute(2, 0, 1)),
        transforms.Lambda(lambda x: F.rotate(x, angle)),
        transforms.Lambda(lambda x: x.permute(1, 2, 0))
      ])
      data = augmentation_data(data)
      data_raw = augmentation_data_raw(data_raw)
      label = augmentation_label(label)

    # Augmentation - Random Resized Crop
    chance = .66
    if self.augment and chance > random.random():
      scale_variability = 0.85
      scale = random.uniform(scale_variability, 1.0)
      scale = .4
      crop_size = int(self.data_size * scale)
      max_offset_x = self.data_size - crop_size
      max_offset_y = self.data_size - crop_size
      offset_x = int(random.randint(0, max_offset_x))
      offset_y = int(random.randint(0, max_offset_y))
      augmentation_data = transforms.Lambda(lambda x: F.resized_crop(x, offset_y, offset_x, crop_size, crop_size, (self.data_size, self.data_size), antialias=None))
      augmentation_data_raw = transforms.Lambda(lambda x: F.resized_crop(x, offset_y, offset_x, crop_size, crop_size, (self.data_size, self.data_size), antialias=None))
      augmentation_label = transforms.Compose([
        transforms.Lambda(lambda x: x.permute(2, 0, 1)),
        transforms.Lambda(lambda x: F.resized_crop(x, offset_y, offset_x, crop_size, crop_size, (self.data_size, self.data_size), antialias=None)),
        transforms.Lambda(lambda x: x.permute(1, 2, 0))
      ])
      data = augmentation_data(data)
      data_raw = augmentation_data_raw(data_raw)
      label = augmentation_label(label)

    return data_raw, data, label