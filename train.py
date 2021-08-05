import os
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dset
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from model import get_model

parser = argparse.ArgumentParser(description='distracted driver detection')
parser.add_argument('--model', default='VGG16', help='type of base model')
parser.add_argument('--transfer', default=False, help='set true if you want to fix pretrained model parameter')
parser.add_argument('--pretrained', default=True, help='set true if you want pretrained model')
parser.add_argument('--lr', default=1e-3, help='learning rate')
parser.add_argument('--epoch', default=30)
parser.add_argument('--batch_size', default=12)
args = parser.parse_args()
            
# --------------------data preprocessing--------------------
# define transform
transform = transforms.Compose([
    transforms.Scale((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# define dataset & data loader
train_dataset = dset.ImageFolder(
    root = './train/',
    transform=transform
)
val_dataset = dset.ImageFolder(
    root='./val',
    transform=transform
)
test_dataset = dset.ImageFolder(
    root = './test',
    transform=transform
)
train_dataloader = DataLoader(
    train_dataset,
    batch_size = args.batch_size,
    shuffle=True
)
val_dataloader = DataLoader(
    val_dataset,
    batch_size = args.batch_size,
    shuffle=False
)
test_Dataloader = DataLoader(
    test_dataset,
    batch_size = args.batch_size,
    shuffle=False
)

# train_features, train_labels = next(iter(train_dataloader))

# --------------------data preprocessing--------------------
# prepare training

model = get_model(args.model, args.transfer, args.pretrained)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=args.lr)
model.cuda()
loss_fn.cuda()

# --------------------model training--------------------
# training
best_acc = 0

for _ in range(args.epoch):
    total_len = 0
    avg_loss = 0
    avg_acc = 0
    val_len = 0
    val_loss = 0
    val_acc = 0

    for image, label in tqdm(train_dataloader):
        image, label = image.cuda(), label.cuda()
        predict = model(image)
        
        loss = loss_fn(predict, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred_y = torch.argmax(predict, dim=1)
        
        total_len += image.shape[0]
        avg_loss += loss.item()
        avg_acc += (pred_y == label).sum().item()
    avg_loss /= total_len
    avg_acc /= total_len

    with torch.no_grad():
        for image, label in val_dataloader:
            image, label = image.cuda(), label.cuda()
            predict = model(image)
            
            loss = loss_fn(predict, label)
            pred_y = torch.argmax(predict, dim=1)

            val_len += image.shape[0]
            val_loss += loss.item()
            val_acc += (pred_y == label).sum().item()
        val_loss /= val_len
        val_acc /= val_len

    if best_acc < val_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), args.model + '_base')

    print(f'accuracy: {avg_acc}, loss: {avg_loss}, val_accuracy: {val_acc}, val_loss: {val_loss}')