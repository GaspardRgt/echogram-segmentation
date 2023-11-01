import os
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# Local imports
from src.builder.loss import semi_supervised_loss
from src.builder.unet import UNet
from src.engine.utils import train
from src.utils.datasetup import SegmentationDataset, create_dataset, custom_sampler, assert_sampler
import src.utils.transforms as segtrans

# Hyperparameters
alpha = 0.03
BATCH_SIZE = 8
C = 10
centers = [[0.01875, 0.05625, -0.01875, -0.01875, -0.01875], [0, 0, 0, 0, 0]]
channels = [5, 16, 32]
class_ids =  [-100, 0, 1, 2, 3, 4, 5, 6, 7]
DEPTH_DIM = False
dataset_name = "ABRACOS2_3F"
data_path = Path("D:/Stage_IRD_2023/semi_supervised_datasets_final")
forcing_alpha = 1
LEARNING_RATE = 1e-3
NUM_WORKERS = 0

# Setting device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Running on following device: {device}")

# Datasets creation
print("\n[INFO] Creating train and validation datasets")

# Set paths
train_data_path = data_path / (dataset_name + "_train") / "dataset"
valid_data_path = data_path / (dataset_name + "_valid") / "dataset"

# Set transforms
if DEPTH_DIM:
    transform = segtrans.Compose([
        segtrans.toZeroOne(max=-40., min=-80.),
        segtrans.ConvertDtype(image_dtype=torch.float32, target_dtype=torch.long),
        segtrans.CorrectClasses(),
        segtrans.MeanRes(depth_dim=DEPTH_DIM),
        segtrans.HorizontalFlip(p=0.5)
    ])
else: # remove channel 0
    transform = segtrans.Compose([
        segtrans.toZeroOne(max=-40., min=-80.),
        segtrans.ConvertDtype(image_dtype=torch.float32, target_dtype=torch.long),
        segtrans.CorrectClasses(),
        segtrans.Remove_channels([0]),
        segtrans.MeanRes(depth_dim=DEPTH_DIM),
        segtrans.HorizontalFlip(p=0.5)
    ])

trainSet, annotated_trainSet = create_dataset(train_data_path,
                                              transforms=transform)

validSet, annotated_validSet = create_dataset(valid_data_path,
                                              transforms=transform)

# Biases sampler creation
print("\n[INFO] Creating biased sampler")
train_sampler = custom_sampler(trainSet, class_ids, p=0.3) # add possibility to state random seed for replicability
assert_sampler(train_sampler, trainSet)

# Dataloaders creation
print("\n[INFO] Creating trainLoader")
trainLoader = torch.utils.data.DataLoader(dataset=trainSet,
                                            batch_size=BATCH_SIZE,
                                            num_workers=NUM_WORKERS,
                                            sampler=train_sampler)

print("\n[INFO] Creating validLoader")
validLoader = torch.utils.data.DataLoader(dataset=validSet,
                                            batch_size=BATCH_SIZE,
                                            num_workers=NUM_WORKERS,
                                            shuffle=False)

# Computing class frequencies and class weights
print("\n[INFO] Computing class weights")
x, _ = trainSet[0]
size = x.shape[-1]
n_samples = len(trainSet) * size**2

Y_tot = []
for _, (_, Y) in tqdm(enumerate(trainLoader)):
    Y_tot.append(Y)
Y_tot = torch.cat(Y_tot)

_, bincount = torch.unique(Y_tot, return_counts=True)
print(f"Bincount for all classes:\n{bincount}")

class_weights = n_samples / (C * bincount)
print(f"Class weights for balanced influence of each class:\n{class_weights}")

# Computing nummber of clusters per class
nb_clusters = np.zeros(8)
for _, (_, Y) in tqdm(enumerate(trainLoader)):
    for y in Y:
        for k in torch.unique(y)[1:]:
            nb_clusters[k] += 1
print(f"Number of clusters in the training set (with biased sampling): {nb_clusters}")

# Instanciating model
print("\n[INFO] Instanciating model")
model = UNet(enc_channels=channels,
             dec_channels=channels[1:][::-1],
             nb_classes=C,
             out_size=(512, 512))

# Creating a loss function and an optimizer
print("\n[INFO] Instanciating loss function and optimizer")
loss_fn = semi_supervised_loss(class_weights,
                               forcing_alpha=forcing_alpha,
                               alpha=alpha,
                               centers=centers,
                               device=device)

optimizer = torch.optim.Adam(model.parameters(),
                             lr=LEARNING_RATE)

from torch.utils.tensorboard import SummaryWriter

from src.engine.utils import train_step

softmax = torch.nn.Softmax(dim=1)

# Create a writer with all default settings
writer = SummaryWriter()

# Train model          
train(model=model.to(device),
      out=softmax,
      train_dataloader=trainLoader,
      valid_dataloader=validLoader,
      optimizer=optimizer,
      loss_fn=loss_fn,
      epochs=1,
      device=device,
      writer=writer)