import os
import csv
import torch
import argparse
import numpy as np
import torchvision.datasets as dset
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from model import get_model

parser = argparse.ArgumentParser(description='distracted driver detection test')
parser.add_argument('--model', default='VGG16', help='type of base model')
parser.add_argument('--model_path', required=True, help='pretrained model path')
parser.add_argument('--save_path', default='./', help='save directory')
args = parser.parse_args()

# --------------------data preprocessing--------------------
# define transform
transform = transforms.Compose([
    transforms.Scale((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_dataset = dset.ImageFolder(
    root = './test',
    transform=transform
)
test_dataloader = DataLoader(
    test_dataset,
    batch_size = 1,
    shuffle=False
)

# define model
model = get_model(args.model)
model.load_state_dict(torch.load(args.model_path))
model.cuda()
model.eval()
print(model)

# --------------------model testing--------------------
result_file = './result.csv'
f = open(result_file, 'w')
file_names = os.listdir('./test/test/')

for idx, image in tqdm(enumerate(test_dataloader)):
    image = image[0].cuda()
    predict = model(image)
    result = predict.detach().cpu().numpy()
    f.write(file_names[idx] + ',')
    for i in range(result.shape[1]):
        f.write(str(result[0][i]) + ',')
    f.write('\n')
