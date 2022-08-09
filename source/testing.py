from preprocess import *
from dataset import FloodNetDataset
from focalloss import FocalLoss
from importlib import reload
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
import os
from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms
import torch.nn as nn
from torchvision.transforms.transforms import ToTensor
import torch.nn.functional as F
import torchvision.models as models
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_deeplab_evaled(
                            path_model: str, 
                            device: str
                      ) -> nn.Module:
    """
    Returns evaled DeepLaV3 model.

    """
    model = models.segmentation.deeplabv3_resnet101(pretrained= False, progress = True, num_classes = 10, pretrained_backbone = False).to(device)
    print(model.load_state_dict(torch.load(path_model, map_location=torch.device(device))))

    model.eval()

    return model 

