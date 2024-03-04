import os
from torch.utils.data import random_split
import torch.nn as nn
import torch.optim as optim
import torch
from torchvision.utils import draw_bounding_boxes, save_image
import torch.utils.data
import torch.nn.functional as F
from torchvision import models, datasets, tv_tensors
from torchvision.transforms import v2
from torch.cuda.memory import memory_allocated, max_memory_allocated, empty_cache
from pycocotools import coco
import matplotlib.pyplot as plt
import cv2
torch.manual_seed(22)
IMAGES_PATH = str('sample_images_to_process')
ANNOTATIONS_PATH = str("sample_images_to_process/result.json")

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"

transforms = v2.Compose(
    [
        v2.ToImage(),
        #v2.RandomPhotometricDistort(p=1),
        #v2.RandomZoomOut(fill={tv_tensors.Image: (123, 117, 104), "others": 0}),
        #v2.RandomIoUCrop(),
        #v2.RandomHorizontalFlip(p=1),
        v2.SanitizeBoundingBoxes(),
        v2.ToDtype(torch.float32, scale=True),
    ])
transforms_raw = v2.Compose(
    [
        v2.ToImage(),
        #v2.RandomPhotometricDistort(p=1),
        #v2.RandomZoomOut(fill={tv_tensors.Image: (123, 117, 104), "others": 0}),
        #v2.RandomIoUCrop(),
        #v2.RandomHorizontalFlip(p=1),
        v2.ColorJitter((0.1,0.1)),
        v2.SanitizeBoundingBoxes(),
    ]
)
dataset_raw = datasets.CocoDetection(IMAGES_PATH, ANNOTATIONS_PATH,transforms=transforms_raw)
dataset_raw = datasets.wrap_dataset_for_transforms_v2(dataset_raw, target_keys=["boxes", "labels"])
dataset = datasets.CocoDetection(IMAGES_PATH, ANNOTATIONS_PATH, transforms=transforms)
dataset = datasets.wrap_dataset_for_transforms_v2(dataset, target_keys=["boxes", "labels"])
dataset_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=1,
    drop_last=False,
    shuffle=False,
    collate_fn=lambda batch: tuple(zip(*batch)),
)
dataset_raw_loader = torch.utils.data.DataLoader(
    dataset_raw,
    batch_size=1,
    drop_last=False,
    shuffle=False,
    collate_fn=lambda batch: tuple(zip(*batch)),
)
model = models.get_model("fasterrcnn_resnet50_fpn_v2", weights=None, weights_backbone=None, num_classes=2).train()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 500
device = torch.device("cuda")
model.to(device)
prev_avg_val_loss = 0

    
def calculate_faster_rcnn_loss(model_output, targets):
    predicted_boxes = model_output[0]['boxes']
    predicted_scores = model_output[0]['scores']
    target_boxes = targets[0]['boxes']
    target_labels = targets[0]['labels']
    if len(target_boxes)<len(predicted_boxes):
        top_k = len(target_boxes)
        predicted_boxes = predicted_boxes[:top_k]
        predicted_scores = predicted_scores[:top_k]
    else:
        top_k = len(predicted_boxes)
        target_boxes = target_boxes[:top_k]
        target_labels = target_labels[:top_k]
    target_labels = target_labels.float()
    regression_loss = F.smooth_l1_loss(predicted_boxes, target_boxes)
    classification_loss = torch.nn.functional.cross_entropy(predicted_scores, target_labels)
    total_loss = classification_loss + regression_loss
    print(f"classification_loss:{classification_loss}, regression_loss={regression_loss}, total_loss={total_loss}")
    return total_loss
for model_dir in os.listdir('saved_models'):
    folder = f"FRCNN_normalised_outputs/{model_dir}"
    model_path = f"saved_models/{model_dir}/trained_model.pth"
    model.load_state_dict(torch.load(model_path))
    model.eval()
    tensor_dict={}
    nr=0
    for images, goals in dataset_raw_loader:
        tensor_dict[nr]=images[0]
        nr += 1
    with torch.no_grad():
        val_loss = 0.0
        nr = 0
        count = 0
        for imgs, targets in dataset_loader:
            imgs = [img.to(device) for img in imgs]
            targets = [{k: v.to(device) for k, v in target.items()} for target in targets]
            loss_dict = model(imgs)
            total_loss = calculate_faster_rcnn_loss(loss_dict, targets)
            val_loss += total_loss
            image_base = tensor_dict[nr]
            result_img = draw_bounding_boxes(tensor_dict[nr],loss_dict[0]['boxes'], fill=True, colors=(255,255,255))
            plt.imshow(  result_img.permute(1, 2, 0)  )
            plt.axis('off')
            output = f"{folder}/{nr}.png"
            os.makedirs(folder, exist_ok=True)
            #plt.savefig(fname=output,format='png',bbox_inches='tight',transparent=True, pad_inches=0)
            nr += 1
        avg_val_loss = val_loss / len(dataset_loader)
