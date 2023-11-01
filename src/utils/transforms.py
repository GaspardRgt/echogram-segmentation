import random

import numpy as np
import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ConvertDtype:
    def __init__(self, image_dtype, target_dtype):
        self.image_dtype = image_dtype
        self.target_dtype = target_dtype

    def __call__(self, image, target):
        image = F.convert_image_dtype(image, self.image_dtype)
        return image, target.to(self.target_dtype)


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target


class Remove38kHz:
    """
    Subtract the 38kHz channel to the others.
    """
    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, image, target):
        C, H, W = image.shape
        image2 = torch.empty(C-1, H, W)
        for c in range(1, C):
            image2[c-1, :, :] = torch.sub(image[c, :, :], self.alpha * image[0, :, :])
        return image2, target


class MeanRes:
    """
    'Mean and Residues': Adds a channel corresponding to the mean, while subtracting it to the other channels.
    Based on known echogram clustering methods using either sum or differences of frequencies.
    """
    def __init__(self, depth_dim):
        self.b = depth_dim

    def __call__(self, image, target):
        C, _, _ = image.shape
        if self.b:
            depth_dim = image[0]
            mean = torch.mean(image[1:], dim=0)
            res = torch.cat([torch.sub(image[c, :, :], mean).unsqueeze(0) for c in range(1, C)])
            return torch.cat([depth_dim.unsqueeze(0), mean.unsqueeze(0), res]), target
        else:
            mean = torch.mean(image, dim=0)
            res = torch.cat([torch.sub(image[c, :, :], mean).unsqueeze(0) for c in range(C)])
            return torch.cat([mean.unsqueeze(0), res]), target


class Remove_channels:
    """
    Removes channels identified by their indexes.
    
    Args:
        idx_list (list): index list of the channels to be removed.
    """
    def __init__(self, idx_list):
        self.idx_list = idx_list

    def __call__(self, image, target):
        C, _, _ = image.shape
        channels_list = []
        for c in range(C):
            if not c in self.idx_list:
                channels_list.append(image[c, :, :].unsqueeze(0))
        return torch.cat(channels_list), target


class toZeroOne:
    """
    Transform data in range (min, max) to range (0, 1).
    """
    def __init__(self, max, min):
        self.max = max
        self.min = min
    
    def __call__(self, image, target):
        depth_dim, img = image[0].unsqueeze(0), image[1:]
        img = (img - self.min)/(self.max - self.min)
        return torch.cat((depth_dim, img), dim=0), target


class CorrectClasses:
    """
    Re-indexes classes starting from 0 (the ground-truth annotations start from 1), while preserving the
    -100 value to be ignored by nn.CrossEntropyLoss.
    """
    def __init__(self):
        pass
    
    def __call__(self, image, target):
        target[target != -100] -= 1
        return image, target
    
    
class HorizontalFlip:
    def __init__(self, p):
        self.p = p

    def __call__(self, image, target):
        if torch.rand(1).item() < self.p:
            return torch.flip(image, [2]), torch.flip(target, [1])
        else:
            return image, target