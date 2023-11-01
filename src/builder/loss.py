import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

# Self-supervised loss

class RFCM_loss(nn.Module):
    def __init__(self, fuzzy_factor=2, regularizer_wt=0.0008, forcing_alpha=1., centers=None, device="cuda"):
        '''
        Unsupervised Robust Fuzzy C-mean loss function for ConvNet based image segmentation
        Junyu Chen, et al. Learning Fuzzy Clustering for SPECT/CT Segmentation
        via Convolutional Neural Networks. Medical physics, 2021 (In press).
        :param fuzzy_factor: exponent for controlling fuzzy overlap, default value = 2
        :param regularizer_wt: weighting parameter for regularization, default value = 0
        Note that ground truth segmentation is NOT needed in this loss fuction, instead, the input image is required.
        :param y_pred: prediction from ConvNet, assuming that SoftMax has been applied.
        :param image: input image to the ConvNet.

        Edit: addded the possibility to "force" the existence of particular clusters, using their centers as input.
        '''
        super().__init__()
        self.fuzzy_factor = fuzzy_factor
        self.wt = regularizer_wt
        self.alpha = forcing_alpha
        self.centers = torch.Tensor(centers)
        self.device = device

    def compute_center(self, mem, img):
        img2 = img.permute(0, 2, 1)
        mem2, img3 = torch.flatten(mem), torch.reshape(img2, (img2.shape[0]*img2.shape[1], img2.shape[2]))
        mem3 = mem2.unsqueeze(1)

        return torch.sum(img3 * mem3, dim=0) / torch.sum(mem3, dim=0, keepdim=True)

    def forward(self, y_pred, image):
        dim = len(list(y_pred.shape)[2:])
        assert dim == 2, 'Supports only 2D!'
        num_clus = y_pred.shape[1]
        ###
        num_channels = image.shape[1]
        pred = torch.reshape(y_pred, (y_pred.shape[0], num_clus, math.prod(list(y_pred.shape)[2:]))) #(bs, Nclust, H*W)
        img = torch.reshape(image, (y_pred.shape[0], num_channels, math.prod(list(image.shape)[2:]))) #(bs, C, H*W)

        kernel = torch.ones((1, 1, 3, 3)).float().to(self.device)
        kernel[:, :, 1, 1] = 0
        ###
        J_1 = 0
        J_2 = 0
        if self.centers is not None:
            n_forced = len(self.centers)
        else:
            n_forced = 0
            
        for k in range(num_clus):
            mem = torch.pow(pred[:, k, ...], self.fuzzy_factor) #extracting membership function (bs, V)
            ###
            if num_clus -1 - k < n_forced:
                v_k = self.centers[num_clus -1 - k].to(self.device)
                J_1 += mem * torch.linalg.norm((img.permute(0, 2, 1) - v_k), ord=2, dim=2) * self.alpha #(bs, V)
            else:
                v_k = self.compute_center(mem, img)
                J_1 += mem * torch.linalg.norm((img.permute(0, 2, 1) - v_k), ord=2, dim=2)
            ###
            J_in = 0
            for j in range(num_clus):
                if k==j:
                    continue
                mem_j = torch.pow(pred[:, j, ...], self.fuzzy_factor)
                ###
                mem_j = torch.reshape(mem_j, (image.shape[0], 1, image.shape[2], image.shape[3]) )
                ###
                res = F.conv2d(mem_j, kernel, padding=int(3 / 2))
                res = torch.reshape(res, (-1, math.prod(list(image.shape)[2:])))
                J_in += res #(bs, V)
            J_2 += mem * J_in #(bs, V)
        return torch.mean(J_1)+self.wt*torch.mean(J_2)

# Semi-supervised loss

class semi_supervised_loss(nn.Module):
    """Semi-supervised loss returning both the self-supervised RFCM loss and supervised nn.CrossEntropyLoss for a given batch.
    The ignore_index argument in nn.CrossEntropyLoss is used to restrain supervision to a few zones of relevant data.
    
    Args:
        class_weights (torch.Tensor): weights compensating class imbalance in nn.CrossEntropyLoss.
    """
    def __init__(self, class_weights, forcing_alpha=1., alpha=0.03, centers=None, device="cuda"):
        super().__init__()
        self.alpha = alpha
        self.RFCM_loss_fn = RFCM_loss(fuzzy_factor=1.,
                                      regularizer_wt=0.,
                                      forcing_alpha=forcing_alpha,
                                      centers=centers,
                                      device=device)
        self.class_weights = class_weights
        self.cross_entropy_fn = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100).to(device)

    def forward(self, y_pred, image, target):
        # Calculate which classes are left "free" by the cross-entropy (the C_free last channels of y_pred)
        C = y_pred.shape[1]
        C_sup = self.class_weights.shape[0]
        C_free = C - C_sup
        
        rfcm_loss = self.RFCM_loss_fn(y_pred, image)
        ce_loss = torch.nan_to_num(self.cross_entropy_fn(y_pred[:, :-C_free, :, :], target), nan=0.)
        total_loss = rfcm_loss + self.alpha * ce_loss
        
        return rfcm_loss, ce_loss, total_loss
