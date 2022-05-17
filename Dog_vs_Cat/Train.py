
import re
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import os
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from efficientnet_pytorch import EfficientNet
import sklearn
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss


class CatDog(Dataset):
    def __init__(self, root):
        self.images = os.listdir(root)
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        file = self.images[index]
        img = np.array(Image.open(os.path.join(self.root, file)))

        if "dog" in file:
            label = 1
        elif "cat" in file:
            label = 0
        else:
            label = -1

        return img, label
    
    

def Get_vectors(model, loader, output_size=(1, 1)):
    model.eval()
    images, labels = [], []

    for idx, (x, y) in enumerate(tqdm(loader)):
        x = x

        with torch.no_grad():
            features = model.extract_features(x)
            features = F.adaptive_avg_pool2d(features, output_size=output_size)
        images.append(features.reshape(x.shape[0], -1).detach().numpy())
        labels.append(y.numpy())
    model.train()
    
    return np.concatenate(images, axis=0) ,  np.concatenate(labels, axis=0)


def train(loader, model, loss_fn, optimizer, scaler):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data
        targets = targets.unsqueeze(1).float()

        with torch.cuda.amp.autocast():
            scores = model(data)
            loss = loss_fn(scores, targets)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()



def main():
    model = EfficientNet.from_pretrained("efficientnet-b7")
    model._fc = nn.Linear(2560, 1)
    train_dataset = CatDog(root="DATA/train/")
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=32,
    )

    model = model
    scaler = torch.cuda.amp.GradScaler()
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    train(train_loader, model, loss_fn, optimizer, scaler)
    images_V, labels_V=Get_vectors(model, train_loader, output_size=(1, 1))
    
    
    X_train, X_val, y_train, y_val = train_test_split(images_V, labels_V, test_size=0.001, random_state=42) 
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    print(f"Accuracy: {clf.score(X_val, y_val)}")
    print(f"LOG LOSS: {log_loss(y_val, val_preds)} ")


if __name__ == "__main__":
    main()