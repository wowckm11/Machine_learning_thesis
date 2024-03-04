import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import os
import shutil
from model import UNET
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)

# Hyperparameters etc.
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1
NUM_EPOCHS = 30
SNAPSHOT_EPOCHS = 900
NUM_WORKERS = 2
IMAGE_HEIGHT = 1200  # 1280 originally
IMAGE_WIDTH = 800 # 1918 originally
PIN_MEMORY = True
LOAD_MODEL = True
TRAIN_IMG_DIR = "images/"
TRAIN_MASK_DIR = "masks/"
VAL_IMG_DIR = "val_images/"
VAL_MASK_DIR = "val_masks/"

def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())


def main():
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )
    flipped_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.HorizontalFlip(p=1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )
    tilted_right_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=(20,25), p=1.0),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )
    tilted_left_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=(-20,-25), p=1.0),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )
    train_5 = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            A.RandomBrightnessContrast(p=1),
            ToTensorV2(),
        ],
    )
    train_6 = A.Compose(
        [
            A.Resize(height=(IMAGE_HEIGHT), width=(IMAGE_WIDTH)),
            A.Rotate(limit=(30,35), p=1.0),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )
    train_7 = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=(40,45), p=1.0),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )
    train_8 = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=(-30,-35), p=1.0),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )
    train_9 = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=(-40,-45), p=1.0),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )
    train_10 = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            A.HorizontalFlip(p=1),
            A.Rotate(limit=(-40,-45), p=1.0),
            ToTensorV2(),
        ],
    )

    val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    def get_all_loaders(loader1,loader2,loader3,loader4,val_loader):
        load_list = [loader1, loader2, loader3, loader4]
        loader_list = []
        for item in load_list:
            
            output, val_loader = get_loaders(
            TRAIN_IMG_DIR,
            TRAIN_MASK_DIR,
            VAL_IMG_DIR,
            VAL_MASK_DIR,
            BATCH_SIZE,
            item,
            val_loader,
            NUM_WORKERS,
            PIN_MEMORY,
        )
            loader_list.append(output)
        return loader_list[0], loader_list[1], loader_list[2], loader_list[3], val_loader
    

    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )
    flipped_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        flipped_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )
    tilted_left_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        tilted_left_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )
    tilted_right_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        tilted_right_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )
    train_loader_5, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_5,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )
    train_loader_6, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_6,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )
    train_loader_7, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_7,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )
    train_loader_8, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_8,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )
    train_loader_9, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_9,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )
    train_loader_10, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_10,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )
    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)


    check_accuracy(val_loader, model, device=DEVICE)
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(NUM_EPOCHS):
        #train 
        #train_fn(train_loader, model, optimizer, loss_fn, scaler)
        #train_fn(flipped_loader, model, optimizer, loss_fn, scaler)
        #train_fn(tilted_left_loader, model, optimizer, loss_fn, scaler)
        #train_fn(tilted_right_loader, model, optimizer, loss_fn, scaler)
        #train_fn(train_loader_5, model, optimizer, loss_fn, scaler)
        #train_fn(train_loader_6, model, optimizer, loss_fn, scaler)
        #train_fn(train_loader_7, model, optimizer, loss_fn, scaler)
        #train_fn(train_loader_8, model, optimizer, loss_fn, scaler)
        #train_fn(train_loader_9, model, optimizer, loss_fn, scaler)
        #rain_fn(train_loader_10, model, optimizer, loss_fn, scaler)

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer":optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)

        # check accuracy
        check_accuracy(val_loader, model, device=DEVICE)

        # print some examples to a folder
        save_predictions_as_imgs(
            val_loader, model, folder="saved_images/", device=DEVICE
        )
        print(f"finished epoch {epoch+1}/{NUM_EPOCHS}")
        if (epoch+1)/SNAPSHOT_EPOCHS in (1,2,3,4,5,6,7,8,9,10):
            snap_name = f'{NUM_EPOCHS}_{(epoch+1)/SNAPSHOT_EPOCHS}_{IMAGE_WIDTH}x{IMAGE_HEIGHT}_UNET_x10aug'
            full_path = f"saved_models/{snap_name}"
            os.makedirs(full_path,exist_ok=True)
            file = f"{full_path}/scores.csv"
            shutil.copyfile('scores.csv',file)
            file = f"{full_path}/my_checkpoint.pth.tar"
            shutil.copyfile('my_checkpoint.pth.tar',file)

            for image in os.listdir('saved_images'):
                file = f"{full_path}/{image}"
                src= f"saved_images/{image}"
                shutil.copyfile(src,file)
            print(f"saved snapshot{epoch/SNAPSHOT_EPOCHS}")


if __name__ == "__main__":
    main()