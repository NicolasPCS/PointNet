from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import numpy as np
import torch.optim as optim
import torch.nn as nn
from data.dataset import ShapeNetDataset
from model.pointnet2 import TNet3d, PointNetClassification
import torch.nn.functional as F
from tqdm import tqdm

def collate_fn(batch_list):
    ret = {}
    ret['class_id'] = torch.from_numpy(np.array([x['class_id'] for x in batch_list])).long()
    ret['class_name'] = np.array([x['class_name'] for x in batch_list])
    ret['points'] = torch.from_numpy(np.stack([x['points'] for x in batch_list], axis=0)).float()
    ret['seg_labels'] = torch.from_numpy(np.stack([x['seg_labels'] for x in batch_list], axis=0)).long
    return ret

def train(model, num_epochs, criterion, optimizer, dataloader_train, label_str = 'class_id', lr_scheduler = None, output_name = 'pointnet_cls.pth', device='cuda'):
    model.to(device)

    for epoch in range(num_epochs):
        print(f"Starting {epoch + 1} epoch ...")

        # Training
        model.train()
        train_loss = 0.0
        for batch_dict in tqdm(dataloader_train, total=len(dataloader_train)):
            # Forward pass
            x = batch_dict['points'].transpose(1, 2).to(device)
            
            labels = batch_dict[label_str].to(device)
            pred = model(x)
            loss = criterion(pred, labels)
            train_loss += loss.item()

            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Adjust learning rate
            if lr_scheduler is not None: 
                lr_scheduler.step()
        
        # Compute per batch losses, metric value
        train_loss = train_loss / len(dataloader_train)

        print(f'EPOCH: {epoch+1}, TRAIN LOSS:{train_loss:6.5f}')
    
    torch.save(model.state_dict(), output_name)


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=2, help="INPUT BATCH SIZE") # 64
parser.add_argument('--num_points', type=int, default=2500, help="NUMBER OF POINTS")
parser.add_argument('--workers', type=int, help="NUMBER OF DATALOADING WORKERS", default=4)
parser.add_argument('--nepoch', type=int, default=5, help="NUMBER OF EPOCHS TO TRAIN FOR")
parser.add_argument('--output_folder', type=str, default='classification', help="OUTPUT FOLDER")
parser.add_argument('--model', type=str, default='', help="MODEL PATH")
parser.add_argument('--dataset_path', type=str, default='/home/ncaytuir/PointNet/data/Shapenetcore_benchmark', help="DATASET PATH")
parser.add_argument('--dataset_type', type=str, default="shapenet", help="DATASET TYPE")
parser.add_argument('--feature_transform', action='store_true', help="USE FEATURE TRANSFORM")
parser.add_argument('--inference', action='store_false', help="INFERENCE")

opt = parser.parse_args()
print(opt)

train_set = ShapeNetDataset(root_dir=opt.dataset_path, split_type='train')
val_set = ShapeNetDataset(root_dir=opt.dataset_path, split_type='val')
test_set = ShapeNetDataset(root_dir=opt.dataset_path, split_type='test')

print(f"Train set length = {len(train_set)}")
print(f"Validation set length = {len(val_set)}")
print(f"Test set length = {len(test_set)}")

# Data loaders
train_loader = torch.utils.data.DataLoader(train_set, batch_size=opt.batch_size, num_workers=opt.workers, shuffle=True, drop_last=True, collate_fn=collate_fn)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=opt.batch_size, num_workers=opt.workers, shuffle=True, drop_last=True, collate_fn=collate_fn)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=opt.batch_size, num_workers=opt.workers, shuffle=True, drop_last=True, collate_fn=collate_fn)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_CLASSES = 16
criterion = nn.NLLLoss()

# Create model, optimizer, lr_scheduler and pass to training fn
classifier = PointNetClassification(NUM_CLASSES)

optimizer = optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)
if torch.cuda.is_available():
    classifier.cuda()

_ = train(model=classifier, num_epochs=opt.nepoch, criterion=criterion, optimizer=optimizer, dataloader_train=train_loader, device=device)
