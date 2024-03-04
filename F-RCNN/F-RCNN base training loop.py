import os
from torch.utils.data import random_split
import torch.nn as nn
import torch.optim as optim
import torch
import torch.utils.data
import torch.nn.functional as F
from torchvision import models, datasets, tv_tensors
from torchvision.transforms import v2
from torch.cuda.memory import memory_allocated, max_memory_allocated, empty_cache
from pycocotools import coco
import shutil

learning = True
IMAGES_PATH = str("coco")
ANNOTATIONS_PATH = str("coco/result.json")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:400"

def load_dataset():
    IMAGES_PATH = str("coco_new_normalised_brightness")
    ANNOTATIONS_PATH = str("coco_new_normalised_brightness/result.json")
    transforms = v2.Compose(
        [
            v2.ToImage(),
            v2.SanitizeBoundingBoxes(),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )


    dataset = datasets.CocoDetection(IMAGES_PATH, ANNOTATIONS_PATH, transforms=transforms)
    dataset = datasets.wrap_dataset_for_transforms_v2(dataset, target_keys=["boxes", "labels"])
    train_size = 105
    val_size = 10
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    return train_dataset, val_dataset
def loader_wrapper(train_dataset, val_dataset):
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        drop_last=True,
        shuffle=True,
        collate_fn=lambda batch: tuple(zip(*batch)),
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        drop_last=False,  
        shuffle=False,  
        collate_fn=lambda batch: tuple(zip(*batch)),
    )
    return train_loader, val_loader
model = models.get_model("fasterrcnn_resnet50_fpn_v2", weights=None, num_classes=2).train()
model.load_state_dict(torch.load('trained_model.pth'))
# Define loss function and optimizer
optimizer = optim.Adam(model.parameters(), lr=0.0005)
# Training loop
device = torch.device("cuda")
model.to(device)

def train_for_epochs(epochs:int, train_loader, val_loader, save=True, mix=True, snapshot=1000):
    prev_avg_val_loss = 0
    epochs_done = 1
    for epoch in range(epochs):
        
        if learning:
            # Training loop
            model.train()
            counter = 0
            batch_loss = {'loss_classifier':0,'loss_box_reg':0, 'loss_objectness':0, 'loss_rpn_box_reg':0,'counter':0}
            for imgs, targets in train_loader:
                counter += 1
                optimizer.zero_grad()
                imgs = [img.to(device) for img in imgs]
                targets = [{k: v.to(device) for k, v in target.items()} for target in targets]
                loss_dict = model(imgs, targets)
                total_loss = sum(loss for loss in loss_dict.values())
                total_loss.backward()
                optimizer.step()
                losses = "" 
                f= open('training_loss.csv', 'a')
                f.write(f"{epoch + 1},{counter},")
                for name, loss_val in loss_dict.items():
                    losses += f"{name} {loss_val:.3f}{'':>20}"
                    batch_loss[name] = batch_loss[name] + float(loss_val)
                    f.write(f"{loss_val:.3f},")
                f.write(f'\n')
                f.close()
                batch_loss['counter'] = batch_loss['counter'] + 1
                print(losses)
                print(f"{memory_allocated()/1000000:.0f}/{max_memory_allocated()/1000000:.0f}MB")
            print(f"Epoch {epoch + 1}/{epochs} - Training completed.")



        # Validation loop

        model.eval()
        def calculate_faster_rcnn_loss(model_output, targets):
            predicted_boxes = model_output[0]['boxes']
            predicted_scores = model_output[0]['scores']
            target_boxes = targets[0]['boxes']
            if len(target_boxes)<len(predicted_boxes):
                top_k = len(target_boxes)
                predicted_boxes = predicted_boxes[:top_k]
                predicted_scores = predicted_scores[:top_k]
            else:
                top_k = len(predicted_boxes)
                target_boxes = target_boxes[:top_k]
            regression_loss = F.smooth_l1_loss(predicted_boxes, target_boxes)
            print(f"regression_loss={regression_loss}")
            return total_loss
        with torch.no_grad():
            val_loss = 0.0
            for imgs, targets in val_loader:
                imgs = [img.to(device) for img in imgs]
                targets = [{k: v.to(device) for k, v in target.items()} for target in targets]
                loss_dict = model(imgs)
                total_loss = calculate_faster_rcnn_loss(loss_dict, targets)
                val_loss += total_loss
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch + 1}/{epochs} - Validation Loss: {avg_val_loss:.3f}")
        #update model if it received higher scores in validation
        if save:
            if prev_avg_val_loss != 0:
                if avg_val_loss < prev_avg_val_loss:
                    torch.save(model.state_dict(), 'trained_model.pth')
                    print("trained model saved")
                    prev_avg_val_loss = avg_val_loss
                else:
                    print("epoch discarded")
            else:
                torch.save(model.state_dict(), 'trained_model.pth')
                print("trained model saved")
                prev_avg_val_loss = avg_val_loss
        #archive model state every x epochs
        if epochs_done/snapshot in (1,2,3,4,5,6,7,8,9,10):
            snap_name = f'{epochs}_{(epoch+1)/snapshot}_FRCN_1x'
            full_path = f"saved_models/{snap_name}"
            os.makedirs(full_path,exist_ok=True)
            file = f"{full_path}/training_loss.csv"
            shutil.copyfile('training_loss.csv',file)
            file = f"{full_path}/trained_model.pth"
            shutil.copyfile('trained_model.pth',file)

            print(f"saved snapshot{epochs_done/snapshot}")

        epochs_done += 1

if __name__ == '__main__':
    train_dataset, val_dataset = load_dataset()
    train_loader, val_loader = loader_wrapper(train_dataset, val_dataset)
    train_for_epochs(50, train_loader,val_loader,False, False, snapshot=10)
