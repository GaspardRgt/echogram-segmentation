import os
from pathlib import Path

import shutil
import argparse

from sklearn.model_selection import train_test_split

"""
Moves a given proportion of the dataset to the test dataset. The rest is used for training and validation.
"""

parser = argparse.ArgumentParser()

parser.add_argument("--dataset_path",
                    default=Path("./data/raw"),
                    help="path from which to retrieve data")

parser.add_argument("--train_size",
                    default=0.9,
                    help="proportion of the dataset to dedicate to training and validation")

args = parser.parse_args()

def main():
    # get data directories paths
    dataset_path = Path(args.dataset_path)
    echo_dir = dataset_path / "echos"
    mask_dir = dataset_path / "masks"
    clusters_dir = dataset_path / "clusters_masks"
    
    imagePaths = [os.path.join(echo_dir, file_path) for file_path in sorted(os.listdir(echo_dir))]
    maskPaths = [os.path.join(mask_dir, file_path) for file_path in sorted(os.listdir(mask_dir))]
    clustersPaths = [os.path.join(clusters_dir, file_path) for file_path in sorted(os.listdir(clusters_dir))]
    
    # create test data directories
    dataset_name = dataset_path.parent.name
    data_path = dataset_path.parent.parent
    test_dataset_path = data_path / (dataset_name +"_test") / "dataset"
    test_echo_dir = test_dataset_path / "echos"
    test_mask_dir = test_dataset_path / "masks"
    test_clusters_dir = test_dataset_path / "clusters_mask"

    # split data betweeen training and validation data
    split = train_test_split(imagePaths,
                             maskPaths,
                             clustersPaths,
                             train_size=float(args.train_size))
    
    (_, testImages) = split[:2]
    (_, testMasks) = split[2:4]
    (_, testClusters) = split[4:]
    
    # move to test dataset directory
    for image_path in testImages:
        image_name = os.path.basename(image_path)
        new_image_path = os.path.join(test_echo_dir, image_name)
        os.makedirs(os.path.dirname(new_image_path), exist_ok=True)
        shutil.move(image_path, new_image_path)
    
    for mask_path in testMasks:
        mask_name = os.path.basename(mask_path)
        new_mask_path = os.path.join(test_mask_dir, mask_name)
        os.makedirs(os.path.dirname(new_mask_path), exist_ok=True)
        shutil.move(mask_path, new_mask_path)
    
    for clusters_path in testClusters:
        patch_name = os.path.basename(clusters_path)
        new_patch_path = os.path.join(test_clusters_dir, patch_name)
        os.makedirs(os.path.dirname(new_patch_path), exist_ok=True)
        shutil.move(clusters_path, new_patch_path)
    
    
    print(f'[INFO] transfered {len(testImages)} patches to test dataset directory.')

if __name__ == "__main__":
    main()