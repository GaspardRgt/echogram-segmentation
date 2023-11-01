# define visualization function
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
from PIL import Image
from shapely.geometry import box
import torch
import torch.nn.functional as F
import torchvision

def plot_grid(Centers, dx:int = 32, dy:int=32, color="gray"):
    """Plots square forming the randomly shifted grid layout.

    Args:
        Centers (list[tuple[int, int]]): list containing the centers of the squares.
        dx (int, optional): squares width. Defaults to 32.
        dy (int, optional): squares height. Defaults to 32.
        color (str, optional): color of the line. Defaults to "gray".
    """
    for (x, y) in Centers:
        box_loc = box(x-dx//2,
                      y-dy//2,
                      x+dx//2,
                      y+dy//2)
        plt.plot(*box_loc.exterior.xy, c=color)

# Visualizing function
def plot_echogram_with_contrast(x, ping_0):
    """Plots part of an echogram.
    
    Args:
        x (torch.Tensor): tensor representing an echogram in CHW format.
        ping_0 (int): ping number of the first column of x. Used for ax legend.
    """
    if x.shape[0] > 3:
        img_tensor = x[:3, :, :]
    else:
        img_tensor = x

    trans = torchvision.transforms.ToPILImage()
    img = trans(img_tensor)

    # img = np.array(x).transpose(1, 2, 0)
    img = plt.imshow(img)

    #plt.colorbar(img, shrink=s)

    plt.yticks([])
    plt.yticks(range(0, x.shape[-2], 200), [int(0.5*i) for i in range(0, x.shape[-2], 200)])

    plt.xticks([])
    plt.xticks(range(0, x.shape[-1], 1000), [(ping_0 + i) for i in range(0, x.shape[-1], 1000)])

    plt.ylabel("depth (m)")
    plt.xlabel("pings")


def add_mean(X, depth_dim=False):
    """Inverse of the custom MeanRes transform. Necessary for correct visualization of data patches.

    Args:
        X (torch.Tensor): NCHW batch of echogram patches.
        depth_dim (bool, optional): weather the first channel of X contains a depths information to conserve.
    Returns:
        torch.Tensor: N(C-1)HW tensor, where X[:, 0, ..] was added to the other channels and deleted.
    """
    X2 = torch.zeros_like(X)
    if depth_dim:
        X2[:, 0, :, :] = X[:, 0, :, :]
        for c in range(2, X.shape[1]):
            X2[:, c-1, :, :] = X[:, c, :, :] + X[:, 1, :, :]
    else:
        mean = x[0]
        for c in range(1, X.shape[1]):
            X2[:, c, :, :] = X[:, c, :, :] + X[:, 0, :, :]
    return X2

def plot_patches(X: list or torch.Tensor, Y: list or torch.Tensor, idx: list, color_list, Preds:list or torch.Tensor=None, color_list2=None):
    """Plots patches with their corresponding masks and possibly the corresponding prediction.
    Echogram patches are displayed in RGB (R: 38kHz, G:70kHz, B:120kHz), the 200kHz channel is ommited.
    
    Args:
        X (torch.Tensor): (N, C, H, W) tensor containing N echogram patches.
        Y (torch.Tensor): (N, H, W) tensor containing ground-truth classes.
        idx (list[int]): list containing the indices of the patches to plot.
        color_list (list[tuple[float, float, float]]): list of RGB colors for the ground-truth masks. Each color is encoded as an RGB tuple, divided by 255.
        Preds (torch.Tensor, optional): (N, Nc, H, W) tensor containing softmax output of the model's predictions on X (Nc is the number of classes). Defaults to None.
        color_list2 (list[tuple[float, float, float]], optional): list of RGB colors for the predicted classes. Defaults to None.
    """
    n_classes_y = len(color_list) 
    if Preds is not None:
        n_classes_preds = Preds.shape[1]
    
    # Creating cmaps
    cmap = colors.ListedColormap(color_list)
    if color_list2 is not None: 
        cmap2 = colors.ListedColormap(color_list2)

    for k in range(len(X)):
        if Preds is not None:
            fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 7))
        else:
            fig, ax = plt.subplots(1, 2, figsize=(10, 7))

        # Plot data RGB
        x = X[k][:3, :, :]
        trans = torchvision.transforms.ToPILImage()
        x_img = trans(x)
        #x_img = np.array(x).transpose(1, 2, 0)
        
        ax[0].imshow(x_img)
        ax[0].set_title(f"Patch", fontsize=15)

        # Plot masks
        # Rearrange data
        y = Y[k]
        #y[y == -100] = 7
        fused_mosaic = y / (n_classes_y - 1)
        y_img = Image.fromarray(np.uint8(cmap(fused_mosaic)[:, :, :-1]*255))
                                     
        # Apply cmap
        ax[1].imshow(y_img)
        ax[1].set_title("Mask", fontsize=15)

        if Preds is not None:
            pred = Preds[k].argmax(dim=0) 
            fused_mosaic = pred / (n_classes_preds - 1)
            pred_img = Image.fromarray(np.uint8(cmap2(fused_mosaic)[:, :, :-1]*255))

            ax[2].imshow(pred_img)
            ax[2].set_title(f"Predictions", fontsize=15)

        fig.suptitle(f"Patch id: {idx[k]}", y=0.85, fontsize=17)


def plot_patches_maps(X: list, Y: list, idx: list, color_list, Preds):
    """Shows one-hot prediction maps for each classes.

    Args:
        X (torch.Tensor): (N, C, H, W) tensor containing N echogram patches.
        Y (list): (N, H, W) tensor containing ground-truth classes.
        idx (list[int]): list containing the indices of the patches to plot.
        color_list (list[tuple[float, float, float]]): list of RGB colors for the ground-truth masks. Each color is encoded as an RGB tuple, divided by 255.
        Preds (torch.Tensor, optional): (N, Nc, H, W) tensor containing softmax output of the model's predictions on X (Nc is the number of classes). Defaults to None.
    """
    n_classes_y = len(color_list)
    cmap = colors.ListedColormap(color_list)
    
    Preds1Hot = F.one_hot(Preds.argmax(dim=1), num_classes=Preds.shape[1]).permute(0, 3, 1, 2).to(torch.float32)
    for k in range(len(X)):
        n = (1 + Preds1Hot.shape[1])//3 + 1
        fig, ax = plt.subplots(nrows=3, ncols=n, figsize=(20, 10))

        # Plot data RGB
        x = X[k][:3, :, :]

        trans = torchvision.transforms.ToPILImage()
        x_img = trans(x)

        ax[0, 0].imshow(x_img)
        ax[0, 0].set_title(f"Echogram patch (3 lowest freqs)", fontsize=10)

        # Plot masks
        # Rearrange data
        y = Y[k]
        fused_mosaic = y / (n_classes_y - 1)
        y_img = Image.fromarray(np.uint8(cmap(fused_mosaic)[:, :, :-1]*255))
           
        # Apply cmap
        ax[0, 1].imshow(y_img)
        ax[0, 1].set_title("Mask (9 classes)", fontsize=10)

        for c in range(Preds1Hot.shape[1]):
            pred = Preds1Hot[k][c]
            pred_img = trans(pred)

            i, j = (2+c) // n,  (2+c) % n
            ax[i, j].imshow(pred_img)
            ax[i, j].set_title(f"Class {c} map", fontsize=10)

        fig.suptitle(f"Patch id: {idx[k]}", y=0.95, fontsize=14)

        #ax[-1, -2].axis('off')
        #ax[-1, -1].axis('off')