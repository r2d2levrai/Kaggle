# -*- coding: utf-8 -*-
import pandas as pd
import torch
from torch.utils.data import TensorDataset
from torch.utils.data.dataset import random_split
from math import ceil
import torch
from sklearn import metrics
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm

def get_data():
    train_data = pd.read_csv("new_train.csv")
    y = train_data["target"]
    X = train_data.drop(["ID_code", "target","has_unique"], axis=1)
    print(X.shape,y.shape)
    X_tensor = torch.tensor(X.values, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.float32)
    ds = TensorDataset(X_tensor, y_tensor)
    train_ds, val_ds = random_split(ds, [int(0.9*len(ds)), ceil(0.1*len(ds))])

    return train_ds, val_ds
train_ds, val_ds = get_data()
train_loader = DataLoader(train_ds, batch_size=1024, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=1024)
#%%
def get_predictions(loader, model, device):
    model.eval()
    saved_preds = []
    true_labels = []

    with torch.no_grad():
        for x,y in loader:
            x = x.to(device)
            y = y.to(device)
            scores = model(x)
            saved_preds += scores.tolist()
            true_labels += y.tolist()

    model.train()
    return saved_preds, true_labels



class NN(nn.Module):
    def __init__(self, input_size, hidden_dim):
        super(NN, self).__init__()
        self.bn = nn.BatchNorm1d(int(input_size/2))
        self.fc1 = nn.Linear(2, hidden_dim) #embbeding layer
        self.fc2 = nn.Linear(int(input_size/2)*hidden_dim, 1)

    def forward(self, x):
        N = x.shape[0]
        x_1=x[:,:200].unsqueeze(2)
        x_2=x[:,200:].unsqueeze(2)
        x = self.bn(x_1)
        x=torch.cat([x_1,x_2],axis=2)
        #print(x.shape)
        x=F.relu(self.fc1(x))
        #print(x.shape)
        x=x.reshape(N,-1)
        x=torch.sigmoid(self.fc2(x))
        x=x.view(-1)
        #print(x.shape)
        return x


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = NN(input_size=400, hidden_dim=100).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
loss_fn = nn.BCELoss()


scaler = torch.cuda.amp.GradScaler()

for epoch in range(10):
    probabilities, true = get_predictions(val_loader, model, device=DEVICE)
    resu=metrics.roc_auc_score(true, probabilities)
    print(f"VALIDATION ROC: {resu}")
    for batch_idx, (data, targets) in tqdm(enumerate(train_loader)):
        data = data.to(DEVICE)
        targets = targets.to(DEVICE)

        # forward

        with torch.cuda.amp.autocast():
            scores = model(data)
            loss = loss_fn(scores, targets)
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    print(loss.detach())


