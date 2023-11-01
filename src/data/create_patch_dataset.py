import os
from pathlib import Path

import argparse
import numpy as np

from src.data import utils

"""
Creates a dataset of square patches extracted from the Echointegration.mat files of one of the ABRACOS
campaigns. The dataset is made of 3 folders. 'echos' contains patches extracted from the data itself, 
'masks' contains a segmentation result used for supervised learning, and 'clusters_masks contains only a few
annotated zones known as regions of interests, used for semi-supervision.

This code is specific to the data folder organization.
"""

parser = argparse.ArgumentParser()

parser.add_argument("--data_path",
                    default=Path("./data/raw"),
                    help="path from which to retrieve data")

parser.add_argument("--dest_path",
                    default=Path("./data/processed"),
                    help="path in which to save formated data")

parser.add_argument("--campaign",
                    type=int,
                    default=1,
                    help="1 for ABRACOS I and 2 for ABRACOS II")

parser.add_argument("--method",
                    type=str,
                    default="3F",
                    help="defines a set of argumments including shape and position of the patches")

args = parser.parse_args()

def listdir_nohidden(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f

def main():
    # Creating parent directory
    DATA_PATH = Path(args.data_path)
    os.chdir(DATA_PATH)

    DEST_PATH = Path(args.dest_path)
    
    # Arguments selection depending on the method:
    method = str(args.method)
    
    if method == "1F":
        depthmin = 0
        depthmax = 1400
        grid_dx = 1200
        grid_dy = 1400
        random_shift = (88, 188)
        grid_offset = (0, 0, 0, 0)
        patch_height = 1024
        patch_width = 1024
        add_depth_dim = True
        n_freqs = 4
        
    elif method == "2F":
        depthmin = 0
        depthmax = 1400
        grid_dx = 1000
        grid_dy = 1000
        random_shift = (20, 20)
        grid_offset = (0, 400, 0, 0)
        patch_height = 960
        patch_width = 960
        add_depth_dim = True
        n_freqs = 4        
    
    elif method == "3F":
        depthmin = 0
        depthmax = 1400
        grid_dx = 550
        grid_dy = 550
        random_shift = (19, 19)
        grid_offset = (0, 850, 0, 0)
        patch_height = 512
        patch_width = 512
        add_depth_dim = True
        n_freqs = 4
        
    elif method == "4F":
        depthmin = 0
        depthmax = 1400
        grid_dx = 348
        grid_dy = 348
        random_shift = (46, 46)
        grid_offset = (0, 1052, 0, 0)
        patch_height = 256
        patch_width = 256
        add_depth_dim = True
        n_freqs = 4
    
    DATASET_PATH = DEST_PATH / f"ABRACOS{int(args.campaign)}_{method}"
    print(f"CREATING PATCH DATASET, saving dataset to {DATASET_PATH}")
    
    # Creating saving directories
    DATASET_PATH = DATASET_PATH / "dataset"
    ECHO_PATH = DATASET_PATH / "echos"
    MASK_PATH = DATASET_PATH / "masks"
    CLUSTER_MASK_PATH = DATASET_PATH / "clusters_masks"

    DATASET_PATH.mkdir(parents=True, exist_ok=True)
    ECHO_PATH.mkdir(parents=True, exist_ok=True)
    MASK_PATH.mkdir(parents=True, exist_ok=True)
    CLUSTER_MASK_PATH.mkdir(parents=True, exist_ok=True)

    # Croping and saving patches from each EI
    for leg in listdir_nohidden(DATA_PATH):
        LEG_PATH = DATA_PATH / leg
        for ei in listdir_nohidden(LEG_PATH):
            EI_PATH = LEG_PATH / ei
            
            # Retrieving data
            Sv_surface, Ind_best_class = utils.getData(dir_path=EI_PATH,
                                                       depthmin=depthmin,
                                                       depthmax=depthmax,
                                                       add_depth_dim=add_depth_dim)
            
            Clusters_mask = np.load(EI_PATH/"Clusters.npy").transpose(1, 0)
            
            Sv_surface = Sv_surface[:int(add_depth_dim) + n_freqs]

            # Creating grid layout
            Centers = utils.layoutRandomGrid(echogram=Sv_surface,
                                             grid_dx=grid_dx,
                                             grid_dy=grid_dy,
                                             shift=random_shift,
                                             offset=grid_offset)
            
            # Saving patches to data and masks directories
            utils.saveSquaresFromGrid(Centers=Centers,
                                      patch_width=patch_width,
                                      patch_height=patch_height,
                                      echogram=Sv_surface,
                                      mask=Ind_best_class,
                                      dataset_dirpath=DATASET_PATH,
                                      ei_path=str(EI_PATH),
                                      clusters_mask=Clusters_mask,
                                      campaign=int(args.campaign))

if __name__ == '__main__':
    main()
