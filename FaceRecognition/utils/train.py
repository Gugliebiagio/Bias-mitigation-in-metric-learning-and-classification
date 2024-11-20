
import torch
import torch.nn as nn
import torch.utils as utils
import os
from tqdm import tqdm
from typing import Callable



def train(
        model: nn.Module,
        train_loader: utils.data.DataLoader,
        device: torch.device,
        optimizer: torch.optim,
        criterion: Callable[[torch.Tensor,torch.Tensor,torch.Tensor],float],
) -> float:
    """trains a neural network for one epoch.
    Args:
        model: the model to train
        train_loader: the data loader containing the training data
        device: the device to use to train the model
        optimizer: the optimizer to use to train the model
        criterion: the loss to optimize
        
    Returns
        the loss value on the training data    """

    loss_train=0
    samples_train = 0
    model.train()
    for images, labels in tqdm(train_loader):
        #TRIPLET DATASET
        
        if isinstance(images,list):
            anchors = images[0].to(device)
            positives = images[1].to(device)
            negatives = images[2].to(device)
            optimizer.zero_grad()
            embedding_anc, embedding_pos, embedding_neg = model(
            anchors, positives, negatives
            )
            
            loss = criterion(embedding_anc, embedding_pos, embedding_neg) 
            
        else:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            
            embedding = model(images)  #modify to use proxy
            loss = criterion(embedding, labels)
        loss_train += loss.item() * len(images)
        samples_train += len(images)
        loss.backward()
        optimizer.step()
        
    loss_train /= samples_train
    return loss_train



# Validate one epoch
def validate(
    model: nn.Module,
    data_loader: utils.data.DataLoader,
    device: torch.device,
    criterion: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], float],
) -> float:
    """Evaluates the model.

    Args:
        model: the model to evalaute.
        data_loader: the data loader containing the validation or test data.
        device: the device to use to evaluate the model.
        criterion: the loss function.

    Returns:
        the loss value on the validation data.
    """
    samples_val = 0
    loss_val = 0.0

    model = model.eval()
    with torch.no_grad():
        for images, labels in tqdm(data_loader):
            
            # TRIPLET
            if isinstance(images, list):
                anchors = images[0].to(device)
                positives = images[1].to(device)
                negatives = images[2].to(device)
                embedding_anc, embedding_pos, embedding_neg = model(
                    anchors, positives, negatives
                )
                loss = criterion(embedding_anc, embedding_pos, embedding_neg)
            else:
                images = images.to(device)
                labels = labels.to(device)
                

                embedding = model(images)
                loss = criterion(embedding, labels)

            loss_val += loss.item() * len(images)
            
            samples_val += len(images)

    loss_val /= samples_val
    return loss_val