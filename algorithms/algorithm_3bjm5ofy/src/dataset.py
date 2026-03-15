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

  def __init__(self, data_dirs, label_dirs, label_input_threshold=25):
    
    self.data_dirs = data_dirs
    self.label_dirs = label_dirs
    self.input_threshold = .1

  def __len__(self):
      return len(self.data_dirs)

  def __getitem__(self, idx):
    # data.shape = (H, W, Cha) = (H, W, 3)
    data = cv2.imread(self.data_dirs[idx], cv2.IMREAD_COLOR)
    data = cv2.resize(data, (455, 256))

    # label.shape = (H, W, Cls) = (H, W, 2)
    label = cv2.imread(self.label_dirs[idx], 0)
    label = cv2.resize(label, (455, 256))
    label_0 = np.zeros(label.shape)
    label_0[label < self.input_threshold] = 1
    label_1 = np.zeros(label.shape)
    label_1[label >= self.input_threshold] = 1
    label = np.stack((label_0, label_1), axis=-1)
    return data, label