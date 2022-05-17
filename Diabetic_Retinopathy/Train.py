import torch
from torch import nn, optim
import os
import Config
from torch.utils.data import DataLoader
from tqdm import tqdm
from efficientnet_pytorch import EfficientNet
from Dataset import DRDataset
from torchvision.utils import save_image



def train(loader, model, optimizer, loss_fn, scaler, device):
    losses = []
    loop = tqdm(loader)
    for batch_idx, (data, targets, _) in enumerate(loop):
        data = data.to(device=device)
        targets = targets.to(device=device)


        with torch.cuda.amp.autocast():
            scores = model(data)
            loss = loss_fn(scores, targets.unsqueeze(1).float())

        losses.append(loss.item())

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        loop.set_postfix(loss=loss.item())

    print(f"Loss in epoch: {sum(losses)/len(losses)}")


def main():
    train_ds = DRDataset(
        images_folder="DATA/train/",
        path_to_csv="DATA/trainLabels.csv",
        transform=Config.val_transforms,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        shuffle=False,
    )

    loss_fn = nn.MSELoss()

    model = EfficientNet.from_pretrained("efficientnet-b3")
    model._fc = nn.Sequential(nn.Linear(1536, 768),
                              nn.BatchNorm1d(768),
                              nn.ReLU(),
                              nn.Dropout(0.2),
                              nn.Linear(768, 100),
                              nn.BatchNorm1d(100),
                              nn.ReLU(),
                              nn.Dropout(0.2),
                              nn.Linear(100, 1),
                              )
    model = model.to(Config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    scaler = torch.cuda.amp.GradScaler()

    

    for epoch in range(Config.NUM_EPOCHS):
        train(train_loader, model, optimizer, loss_fn, scaler, Config.DEVICE)

        if epoch % 2==0:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            torch.save(checkpoint, f"b3_{epoch}.pth.tar")



if __name__ == "__main__":
    main()
