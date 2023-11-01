import os
import random
import h5py
import torch
import numpy as np
from pathlib import PosixPath
import pickle
from typing import Union

def ApplyThresholds(x: torch.Tensor(), low:torch.float = -80., high: int = None):
    """Applies a threshold on a tensor by limiting values to a minimum and a maximum. 
    Also assigns NaN values to low. Returns a new tensor with thresholded values.
    
    Args:
        x (torch.Tensor): a PyTorch tensor.
        low (torch.float,  optional): the minimum value to which any value below will be assigned. Defaults to -80.
        high (int, optional): the maximum value to which any value above will be assigned. Defaults to None.
    
    Returns: 
        torch.Tensor: Tensor with values s
    """
    low_tensor = torch.full(x.shape, fill_value=low)
    if high:
        high_tensor = torch.full(x.shape, fill_value=high)
        return torch.minimum(torch.maximum(torch.nan_to_num(x, nan=low),
                                           low_tensor),
                             high_tensor)
    else:
        return torch.maximum(torch.nan_to_num(x, nan=low),
                             low_tensor)

def ApplyNoiseDepthLim(x: torch.Tensor(), depth_limits: list, low: torch.float = -80.):
    """For each channel in the echointegration tensor (CHW), sets pixels below the given depth limit to a given value.
    This intends to suppress the remaining noise for each frequency in the echogram.

    Args:
        x (torch.Tensor): a PyTorch tensor in CHW format representing echointegration data.
        depth_limits (list): list containing the depths below which each channel should have their values set to low.
        low (torch.float, optional): value assigned to pixels below depth limit. Defaults to -80..

    Returns:
        torch.Tensor: input with no noise left below depth limits.
    """
    for c in range(len(depth_limits)):
        x[c, depth_limits[c]:, :] = low
    return x

# Retrieving data and returning in the right format
def getData(dir_path: PosixPath,
            depthmax: int,
            depthmin: int = 0,
            add_depth_dim: bool = False):
    """Retrieves HDF5 echointegration and segmentation files from directory. Crops the depth of the data.
    Returns data with a few modifications for further use in segmentation.

    Args:
        dir_path (PosixPath): the path to the directory in which the HDF5 are located.
        depthmax (int): the maximum depth to be kept when croping data.
        depthmin (int, optional): the minimum depth to be kept when croping data. Defaults to 0.
        add_depth_dim (bool, optional): whether or not to add a channel with values proportional to depth. Defaults to False.

    Raises:
        Exception: if unable to load echointegration file.
        Exception: if unable to load segmentation file.

    Returns:
        Sv_surface (torch.Tensor): CHW float tensor containing pre-processed back-scattering (Sv relative to the surface) values of the echointegration file.
        Ind_best_class_wFISH (torch.Tensor): CHW float tensor containing the labels for each in Sv_surface.
    """
    
    print(f"\n**********Retrieving data from: {dir_path} ...**********")
              
    # Reading files
    for filename in os.listdir(dir_path):
        if filename == "Echointegration.mat":
            EI = h5py.File(dir_path/filename)
        elif filename.split("_")[-1] == "6groups.mat" or filename.split("_")[-1] == "7groups.mat":
            CLASSES = h5py.File(dir_path/filename)
        else: ()
         
    try:
        EI.keys()
    except:
        raise Exception(f"Issue while retrieving EI file in {dir_path}")
    try:
        CLASSES.keys()
    except:
        raise Exception(f"Issue while retrieving CLASSES file in {dir_path}")
    print(f"Files retrieved:\n    {EI}\n    {CLASSES}")
    
    # Getting matrixes from HDF5 files, converting to tensors and correct format
    # Sv Data
    Sv_surface = torch.nan_to_num(torch.from_numpy(np.array(EI.get('Sv_surface'))).permute(0, 2, 1)[:, depthmin:depthmax, :],
                              nan=-80.)
    Sv_surface = ApplyThresholds(Sv_surface, low=-80., high=-40.)
    
    Sv_surface = ApplyNoiseDepthLim(Sv_surface, depth_limits=[1400, 1000, 550, 300], low=-80.)
    
    # adding depth channel
    if add_depth_dim:
        print("Adding depth channel...")
        _, H, W = Sv_surface.shape
        depth_values = torch.arange(0, H, dtype=torch.float) / H
        depth_dim = depth_values.unsqueeze(0).repeat(1, W, 1).permute(0, 2, 1)
        Sv_surface = torch.cat((depth_dim, Sv_surface), dim=0)

    # Label
    Ind_best_class = torch.nan_to_num(input=torch.from_numpy(np.array(CLASSES.get('ind_Best_class'))).permute(1, 0)[depthmin:depthmax, :],
                                      nan=-2.)
    Ind_best_class = Ind_best_class.to(torch.long)
    
    n = int(torch.max(Ind_best_class))
    Mask_FISH_2D = torch.nan_to_num(torch.from_numpy(np.array(EI.get('Mask_clean_FISH'))).permute(0, 2, 1)[0, depthmin:depthmax, :], 
                            nan=0.).to(torch.long)
    Ind_best_class_wFISH = Ind_best_class.masked_fill_(Mask_FISH_2D==1, n+1) # adding SVS to the classification
    
    EI.close()
    CLASSES.close()

    if add_depth_dim:
        print(f"Sv_surface tensor shape: {Sv_surface.shape} -> [depth + color_channels (1+4), height, width] (CHW)")
    else:
         print(f"Sv_surface tensor shape: {Sv_surface.shape} -> [color_channels (4), height, width] (CHW)")

    return Sv_surface, Ind_best_class_wFISH

def layoutRandomGrid(echogram: torch.Tensor(), 
                     grid_dx: int, 
                     grid_dy: int,
                     shift:int,
                     offset: tuple = (0, 0, 0, 0)):
    """Creates a grid layout for a CHW tensor representing an echogram, with rectangular patches of given dimensions.

    Args:
        echogram (torch.Tensor): the CHW echogram tensor on which we the grid should be laid out.
        grid_dx (int): the mean interval between patches centers along the x axis.
        grid_dy (int): the mean interval between patches centers along the x axis.
        shift (int): range of the random shift applied to each center.
        offset (tuple, optional): the offset between the edges of the grid and those of the image, written (top, bottom, left, right). Defaults to (0, 0, 0, 0).

    Returns:
        Centers (list): list of the centers of the patches making the grid.
    """
    
    _, H, W = echogram.shape
    k_vertical = (H - offset[0] - offset[1]) // grid_dy
    k_horizontal = (W - offset[2] - offset[3]) // grid_dx
    
    Centers = []

    for i in range(0, k_vertical):
        y = offset[0] + int((i+1/2) * grid_dy)
        for j in range(0, k_horizontal):
            x = offset[2] + int((j+1/2) * grid_dx)
            if type(shift) == tuple:
                xshift, yshift = shift
                Centers.append((x + random.randint(-xshift, xshift), y + random.randint(-yshift, yshift)))
            else:
                Centers.append((x + random.randint(-shift, shift), y + random.randint(-shift, shift)))
    
    return Centers


# crop and save a copy of the data in dataset directory. For maks, apply a custom torch.transforms to obtain a regular format mask
def cropNcopy(image: Union[torch.Tensor, np.ndarray], center:tuple, dx:int, dy:int, transform=None):
    """Returns a patch croped from the input image.
    Args:
        image (Union[torch.Tensor, np.ndarray]): image from which to crop a patch.
        center (tuple): coordinate of the center of the patch in the imput image
        dx (int): width of the patch.
        dy (int): height of the patch.
        transform (optional): transform to apply to the patch before returning. Defaults to None.

    Returns:
        Union[torch.Tensor, np.ndarray]: patch croped from the input image.
    """ 
    x, y = center
    ndim = len(image.shape)
    assert ndim == 2 or ndim == 3, "ndmin should be 2 or 3"
    if ndim == 2:
        if transform:
            return transform(image[y-dy//2:y+dy//2, x-dx//2:x+dx//2])
        else:
            return image[y-dy//2:y+dy//2, x-dx//2:x+dx//2]
    else:
        if transform:
            return transform(image[:, y-dy//2:y+dy//2, x-dx//2:x+dx//2])
        else:
            return image[:, y-dy//2:y+dy//2, x-dx//2:x+dx//2]


def saveSquaresFromGrid(Centers:list,
                        patch_width:int,
                        patch_height:int, 
                        echogram:torch.Tensor,
                        mask:torch.Tensor,
                        dataset_dirpath: PosixPath,
                        ei_path: PosixPath,
                        campaign:int,
                        clusters_mask:np.ndarray=None):
    """Saves patches croped from an echointegration tensor and the coresponding mask tensor.

    Args:
        Centers (list): list of the coordinates (tuple) for the centers of the patches.
        patch_width (int): width of the patches.
        patch_height (int): height of the patches.
        echogram (torch.Tensor): CHW tensor representing the echogram.
        mask (torch.Tensor): HW tensor representing a mask
        dataset_dirpath (PosixPath): path of the directory in which to save the patches folders.
        ei_path (PosixPath): path of the original echointegration file directory.
        campaign (int): 1 for ABRACOS I and 2 for ABRACOS II.
        clusters_mask (np.ndarray, optional): HW masks representing a few important pixel clusters. Defaults to None.
    """
    leg, ei = ei_path.split("\\")[-2], ei_path.split("\\")[-1]
    save_dir_data = dataset_dirpath / "echos"
    save_dir_masks = dataset_dirpath / "masks"
    save_dir_clusters_masks = dataset_dirpath / "clusters_masks"

    Centers_dict = {}

    print(f"Saving patches from {ei} of {leg}...")
    for i in range(len(Centers)):

            save_path_data = save_dir_data / f"ABRACOS{campaign}_{leg}_{ei}_echopatch_{i}.npy"
            save_path_masks = save_dir_masks / f"ABRACOS{campaign}_{leg}_{ei}_maskpatch_{i}.npy"
            save_path_clusters_masks = save_dir_clusters_masks / f"ABRACOS{campaign}_{leg}_{ei}_clusterpatch_{i}.npy"
            

            center = Centers[i]
            
            Centers_dict[f"echopatch_{i}"] = center
    
            # Saving data
            try:
                image_patch = cropNcopy(image=echogram,
                                        center=center, 
                                        dx=patch_width,
                                        dy=patch_height).numpy()
                np.save(save_path_data,
                        image_patch)
                
                mask_patch = cropNcopy(image=mask,
                                       center=center,
                                       dx=patch_width,
                                       dy=patch_height).numpy()
                
                np.save(save_path_masks, 
                        mask_patch)
                
                if clusters_mask is not None:
                    cluster_patch = cropNcopy(image=clusters_mask,
                                              center=center,
                                              dx=patch_width,
                                              dy=patch_height)
                    
                    np.save(save_path_clusters_masks, 
                            cluster_patch)
                        
                    
            except Exception as e:
                print(f"Error while saving ecopatch {i}: {e}")
    
    dict_path = dataset_dirpath / f'centers_dict_ABRACOS{campaign}_{leg}_{ei}.pickle'
    with open(dict_path, 'wb') as f:
        pickle.dump(Centers_dict, f)