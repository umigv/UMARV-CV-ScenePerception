import sys
import numpy as np
import cv2
import sys
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import glob
import random

class Dataset_Class(Dataset):

  # def __init__(self, device, split, datasets, transform, data_augmentation, data_augmentation_weight):
  def __init__(self, data_dirs, label_dirs, device=None, label_input_threshold=None):

    if device is None:
      device = 'cpu'
    if label_input_threshold is None:
      label_input_threshold = .1
    
    self.data_dirs = data_dirs
    self.label_dirs = label_dirs
    self.device = device
    self.input_threshold = .1

    #Initialize default transforms
    self.default_data_transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Resize((128, 128), antialias=None)
    ])
    self.default_label_transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Resize((128, 128), antialias=None),
      transforms.Grayscale(1)
    ])

    # #Initialize custom transforms
    # self.transform = transform
    # self.data_augmentation = data_augmentation
    # self.data_augmentation_weight = data_augmentation_weight

    # #Check if data augmentation list and weight list are same length
    # if len(data_augmentation) != len(data_augmentation_weight):
    #   print(f'Error : The lengths of data_augmentation ({len(data_augmentation)}) and data_augmentation_weight ({len(data_augmentation_weight)}) do not match')
    #   sys.exit()

  def __len__(self):
      return len(self.data_dirs)

  def __getitem__(self, idx):
    # Get data - data.shape=torch.Size([128, 128, 3])
    data = cv2.imread(self.data_dirs[idx], cv2.IMREAD_COLOR)
    data = self.default_data_transform(data)
    data = data.to(self.device)
    data_raw = data.detach().clone()
    # if self.transform:
    #   data = self.transform(data)
    # if len(self.data_augmentation) > 0 and self.split == 'train':
    #   chance_of_augmentation = .5
    #   if chance_of_augmentation > random.random():
    #     idx = random.choices(np.arange(len(self.data_augmentation)), weights=self.data_augmentation_weight, k=1)[0]
    #     augmentation = self.data_augmentation[idx]
    #     data = augmentation(data)

    #Get label - label.shape=torch.Size([128, 128, 2])
    label = cv2.imread(self.label_dirs[idx])
    label = self.default_label_transform(label)
    label = label.to(self.device)
    label_0 = torch.zeros(label.shape, device=self.device)
    label_0[label < self.input_threshold] = 1
    label_1 = torch.zeros(label.shape, device=self.device)
    label_1[label >= self.input_threshold] = 1
    label = torch.stack((label_0.squeeze(), label_1.squeeze()))

    return data_raw, data, label