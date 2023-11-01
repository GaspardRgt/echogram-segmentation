import os

import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, Dataset


# create custom Dataset class
class SegmentationDataset(Dataset):
    """
    Custom segmentation dataset class inheriting torch.utils.data.Dataset
    """
    def __init__(self, imagePaths, maskPaths, transforms):
        # store the image and mask filepaths, and augmentation transforms
        self.imagePaths = imagePaths
        self.maskPaths = maskPaths
        self.transforms = transforms
    
    def __len__(self):
        # return the number of total samples contained in the dataset
        return len(self.imagePaths)
    
    def __getitem__(self, idx):
        # grab the image path from the current index
        image = torch.from_numpy(np.load(self.imagePaths[idx]))

        # read the associated masl from disk
        mask = torch.from_numpy(np.load(self.maskPaths[idx]))

        # apply transformations if necessary
        if self.transforms is not None:
            image, mask = self.transforms(image, mask) # using PyTorch reference segmentation transforms (https://github.com/pytorch/vision/blob/main/references/segmentation/transforms.py)
        
        # return a tuple
        return (image, mask)

def create_datasets(image_dir, mask_dir, train_size, transforms):
    """Creates datasets for training and validation

    Args:
        image_dir (PosixPath or str): dataset path.
        mask_dir (PosixPath or str): ground truth dataset path.
        train_size (float): _description_
        transforms (torchvision.Transform): transforms to apply to segmentation data.

    Returns:
        tuple[SegmentationDataset, SegmentationDataset]: training and validation datasets.
    """
    # load the image and mask filepaths in a sorted manner
    imagePaths = [os.path.join(image_dir, file_path) for file_path in sorted(os.listdir(image_dir))]
    maskPaths = [os.path.join(mask_dir, file_path) for file_path in sorted(os.listdir(mask_dir))]

    # split data betweeen training and validation data
    split = train_test_split(imagePaths,
                            maskPaths,
                            train_size=train_size)

    (trainImages, validImages) = split[:2]
    (trainMasks, validMasks) = split[2:]

    # instanciate train and validation datasets
    trainDataset = SegmentationDataset(imagePaths=trainImages,
                                    maskPaths=trainMasks,
                                    transforms=transforms)
    validDataset = SegmentationDataset(imagePaths=validImages,
                                    maskPaths=validMasks,
                                    transforms=transforms)

    print(f"\n[INFO] found {len(trainDataset)} examples in the training set...")
    print(f"[INFO] found {len(validDataset)} examples in the validation set...")

    return trainDataset, validDataset

def create_dataloaders(image_dir, mask_dir, train_size, transforms, batch_size, num_workers):
    """Create dataloaders for training and validation.

    Args:
        image_dir (PosixPath or str): dataset path.
        mask_dir (PosixPath or str): ground truth dataset path.
        train_size (float): _description_
        transforms (torchvision.Transform): transforms to apply to segmentation data.
        batch_size (int): dataloader batch size.
        num_workers (int): dataloader number of workers.

    Returns:
        tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]: training and validation
    """
    # instanciate train and validation datasets
    trainDataset, validDataset = create_datasets(image_dir, mask_dir, train_size, transforms)

    print(f"\n[INFO] found {len(trainDataset)} examples in the training set...")
    print(f"[INFO] found {len(validDataset)} examples in the validation set...\n")

    # create train and validation dataloaders from datasets

    trainLoader = DataLoader(dataset=trainDataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=num_workers)
    validLoader = DataLoader(dataset=validDataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers)
    
    return trainLoader, validLoader

def create_test_dataloader(image_dir, mask_dir, transforms, batch_size, num_workers):
    """Creates dataloader for testing.

    Args:
        image_dir (PosixPath or str): dataset path.
        mask_dir (PosixPath or str): ground truth dataset path.
        transforms (torchvision.Transform): transforms to apply to segmentation data.
        batch_size (int): dataloader batch size.
        num_workers (int): dataloader number of workers.

    Returns:
        tuple[SegmentationDataset, torch.utils.data.DataLoader]: test dataset and corresponding test dataloader
    """
    # load the image and mask filepaths in a sorted manner
    imagePaths = [os.path.join(image_dir, file_path) for file_path in sorted(os.listdir(image_dir))]
    maskPaths = [os.path.join(mask_dir, file_path) for file_path in sorted(os.listdir(mask_dir))]

    # instanciate train and validation datasets
    testDataset = SegmentationDataset(imagePaths=imagePaths,
                                      maskPaths=maskPaths,
                                      transforms=transforms)

    print(f"\n[INFO] found {len(testDataset)} examples in the testing set...")

    # create train and validation dataloaders from datasets

    testLoader = DataLoader(dataset=testDataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers)
    
    return testDataset, testLoader

#########################################################################
##################### New dataset creation function #####################
#########################################################################

def create_dataset(data_path, transforms):
    
    images_dir = data_path/"echos"
    masks_dir = data_path/"masks"
    clusters_dir = data_path/"clusters_masks_wFISH"
    
    imagePaths = [os.path.join(images_dir, file_path) for file_path in sorted(os.listdir(images_dir))]
    maskPaths = [os.path.join(masks_dir, file_path) for file_path in sorted(os.listdir(masks_dir))]
    clusterPaths = [os.path.join(clusters_dir, file_path) for file_path in sorted(os.listdir(clusters_dir))]

    Dataset = SegmentationDataset(imagePaths,
                                  clusterPaths,
                                  transforms=transforms)
    
    Vargas_segmentation_Dataset = SegmentationDataset(imagePaths,
                                                      maskPaths,
                                                      transforms=transforms)
                        
    return Dataset, Vargas_segmentation_Dataset


# create sampler to sample patches with ground truth with a probability p

import numpy as np
import random
from tqdm import tqdm

def get_indices(dataset, class_ids):
    """Collect patches indices from dataset and group them depending on the classes represented on those patches.

    Args:
        dataset (torch.utils.data.Dataset): torch Dataset from which to collect information.
        class_ids (list): list containing the integer ids for each class.

    Returns:
        dict: dictionary with class ids as keys and patches indices as values.
    """
    print("Collecting indices by class from dataset...")
    # Collect indices of samples for each class
    indices_by_class = {class_id: [] for class_id in class_ids}
    for idx in tqdm(range(len(dataset))):
        _, y = dataset[idx]
        class_ids = torch.unique(y)  # Get unique class IDs in annotation
        if len(class_ids) > 1:  
            for class_id in class_ids[1:]: 
                indices_by_class[class_id.item()].append(idx)
        else:
            indices_by_class[-100].append(idx) # Make sure to separate patches without annotations from the rest

    return indices_by_class

def custom_sampler(dataset, class_ids, p):
    """Custom sampler ensuring a sufficient proportion of annotated patches (through up-sampling) for semi-supervised learning.

    Args:
        dataset (torch.utils.data.Dataset): Dataset from which to create dataloader.
        class_ids (list): list containing the integer ids for each class.
        p (float): desired proportion of annotated patches.

    Returns:
        list: list of indices of the patches sampled for dataloader.
    """
    indices_by_class = get_indices(dataset, class_ids)
    n_samples = len(dataset)

    print(f"\nCreating biased sampler with {100*p:.2f}% of annotated patches...")
    num_classes = len(indices_by_class.keys())
    
    counts = [len(class_indices) for class_indices in indices_by_class.values()][1:]
    probabilities = counts / np.sum(counts)

    num_samples_by_class = {k: 0 for k in indices_by_class.keys()}
    for n in range(n_samples):
        proba = np.random.rand()
        if proba <= p:
            k = np.random.choice(np.arange(num_classes-1), size=1, p=probabilities).item()
            num_samples_by_class[k] += 1
        else:
            num_samples_by_class[-100] += 1

    sampled_indices = []
    for k in indices_by_class.keys():
        class_indices = indices_by_class[k]
        sampled = [class_indices[i] for i in torch.randperm(len(class_indices))[:num_samples_by_class[k]]]
        
        if k == - 100:
            sampled = [class_indices[i] for i in np.random.choice(len(class_indices), num_samples_by_class[k], replace=False)]
        else:
            sampled = [class_indices[i] for i in np.random.choice(len(class_indices), num_samples_by_class[k], replace=True)]

        sampled_indices.extend(sampled)

    random.shuffle(sampled_indices)
    print("Sampler created.")

    return sampled_indices

def assert_sampler(sampled, dataset):
    """
    Asserts and prints the proportion of annotated patches in the sampled list.

    Args:
        sampled (list): list of sampled patches indices.
        dataset (torch.utils.data.Dataset): dataset from which patches were sampled.
    """
    print("Asserting sampler...")
    n = 0
    for class_id in tqdm(sampled):
        _, y = dataset[class_id]
        if len(torch.unique(y)) > 1:
            n +=1

    prop = 100*n/len(sampled)
    print(f"\nProportion of annotated patches in sample over whole trainSet: {prop:.2f}")