import torch 
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image,ImageOps
import numpy as np
from torchvision.transforms import transforms
from torch.utils.data import Dataset,DataLoader
import torchvision.transforms.functional as F
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
import os
import torch.optim as optim


def train(train_loader ,model,epochs = 5, lr = 0.001):
    criterion =  nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(epochs):
        running_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            # Forward pass
            outputs = model(inputs)

            # Compute loss
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # Print average loss for the epoch
        average_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {average_loss}")

    print("Training finished!")

def eval(model,test_loader):
    num_samples = 0
    num_correct = 0
    num_pixels = 0
    dice_score = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            predictions = (outputs > 0).float()

            num_correct += (predictions==targets).sum()
            num_pixels += torch.numel(predictions)
            dice_score += (2*(predictions*targets).sum()/((predictions+targets).sum()+1e-8))

            num_samples += inputs.size(0)

        print(f'Got {num_correct}/{num_pixels} with accuracy {num_correct/num_pixels*100:.2f} %')
        print(f"Dice score: {dice_score/len(test_loader)}")