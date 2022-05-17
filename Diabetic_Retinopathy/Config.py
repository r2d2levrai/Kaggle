import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from Dataset import DRDataset
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from PIL import Image
import cv2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 3e-5
WEIGHT_DECAY = 5e-4
BATCH_SIZE = 20
NUM_EPOCHS = 100
NUM_WORKERS = 6
CHECKPOINT_FILE = "b3.pth.tar"
PIN_MEMORY = True
SAVE_MODEL = True
LOAD_MODEL = True

def trim(im):
    """
    Converts image to grayscale using cv2, then computes binary matrix
    of the pixels that are above a certain threshold, then takes out
    the first row where a certain percetage of the pixels are above the
    threshold will be the first clip point. Same idea for col, max row, max col.
    """
    percentage = 0.02

    img = np.array(im)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    im = img_gray > 0.1 * np.mean(img_gray[img_gray != 0])
    row_sums = np.sum(im, axis=1)
    col_sums = np.sum(im, axis=0)
    rows = np.where(row_sums > img.shape[1] * percentage)[0]
    cols = np.where(col_sums > img.shape[0] * percentage)[0]
    min_row, min_col = np.min(rows), np.min(cols)
    max_row, max_col = np.max(rows), np.max(cols)
    im_crop = img[min_row : max_row + 1, min_col : max_col + 1]
    return Image.fromarray(im_crop)

def get_mean_std(loader):
    channels_sum,channels_squared_sum,num_batches=0,0,0
    for data,_ in loader:
        channels_sum+=torch.mean(data,dim=[0,2,3])
        channels_squared_sum+=torch.mean(data**2,dim=[0,2,3])
        num_batches+=1
    mean=channels_sum/num_batches
    std=(channels_squared_sum/num_batches -mean**2)**0.5
    
    return mean,std

train_transforms = A.Compose(
    [
        A.Resize(width=760, height=760),
        A.RandomCrop(height=728, width=728),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Blur(p=0.3),
        A.CLAHE(p=0.3),
        A.ColorJitter(p=0.3),
        A.CoarseDropout(max_holes=12, max_height=20, max_width=20, p=0.3),
        A.IAAAffine(shear=30, rotate=0, p=0.2, mode="constant"),
        A.Normalize(
            mean=[0.3199, 0.2240, 0.1609],
            std=[0.3020, 0.2183, 0.1741],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ]
)

val_transforms = A.Compose(
    [
        A.Resize(height=728, width=728),
        A.Normalize(
            mean=[0.3199, 0.2240, 0.1609],
            std=[0.3020, 0.2183, 0.1741],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ]
)


if __name__ == "__main__":
    dataset = DRDataset(
        images_folder="DATA/resized_train_cropped/",
        path_to_csv="DATA/trainLabels_cropped.csv",
        transform=val_transforms,
    )
    loader = DataLoader(
        dataset=dataset, batch_size=32, num_workers=2, shuffle=True, pin_memory=True
    )    
    
    print(get_mean_std(loader))