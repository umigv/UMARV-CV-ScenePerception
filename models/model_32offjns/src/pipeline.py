from ultralytics import YOLO
from matplotlib import pyplot as plt
import os
import numpy as np
import torch
from methods import get_label, get_img, upload_dataset_to_dropbox
from architecture import lane_model

# TODO: assign dbx token from env here
DBX_TOKEN = ...
threshold_conf = 0.85

## Helper functions

def get_label(result):
    masks = result.masks.data
    # Sum the masks and use a threshold to create a binary mask
    combined_mask = torch.sum(masks, dim=0) > 0

    # Convert to numpy
    combined_mask_np = combined_mask.cpu().numpy().astype(int)

    return combined_mask_np

def get_img(result):
    return result.orig_img
##############################################

def run_pipeline(model_wts_path, img_dataset_path):
    model = YOLO(model_wts_path)
    
    # List all image files in the directory
    image_files = [os.path.join(img_dataset_path, img) for img in os.listdir(img_dataset_path) if img.endswith(('.png', '.jpg', '.jpeg'))]

    results = model(image_files)
    new_train_imgs = []
    new_train_lbls = []

    for result in results:
        if np.mean(result.boxes.conf) > threshold_conf:
            new_train_imgs.append(get_img(result))
            new_train_lbls.append(get_label(result))
    
    # save new training images and labels to "YOLO_soft_labeled_data" folder
    for i, (img, lbl) in enumerate(zip(new_train_imgs, new_train_lbls)):
        plt.imsave(f"YOLO_soft_labeled_data/img_{i}.png", img)
        np.save(f"YOLO_soft_labeled_data/lbl_{i}.npy", lbl)
    
    upload_dataset_to_dropbox("YOLO_soft_labeled_data", DBX_TOKEN)
################################################

## Training loop functions

def get_acc(model, loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

def train(train_loader, val_loader, criterion=torch.nn.CrossEntropyLoss()):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = lane_model(lookback={'count': 3}).to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

    epochs = 10
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        if epoch % 10 == 0:
            print(f'Epoch {epoch} loss: {running_loss}')
            print("Train Acc: ", get_acc(model, train_loader))
            print("Val Acc: ", get_acc(model, val_loader))
            print("\n")
    return model