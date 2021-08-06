import os
import torch
import argparse
import torchvision.datasets as dset
from model import get_model
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

parser = argparse.ArgumentParser(description='distracted driver detection test')
parser.add_argument('--model', default='VGG16', help='type of base model')
parser.add_argument('--transfer', default=False, help='set true if you want to fix pretrained model parameter')
parser.add_argument('--pretrained', default=True, help='set true if you want pretrained model')
parser.add_argument('--model_path', default='pretrained_model/VGG16_base', help='pretrained model path')
args = parser.parse_args()

# define dataset to find abnormal data
transform = transforms.Compose([
    transforms.Scale((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = dset.ImageFolder(
    root = './train',
    transform=transform
)
train_dataloader = DataLoader(
    train_dataset,
    batch_size = 1,
    shuffle=False
)

# define model
model = get_model(args)
model.load_state_dict(torch.load(args.model_path))
model.cuda()
model.eval()

# find abnormal data
result_file = './pretrained_model/abnormal.csv'
f = open(result_file, 'w')
file_names = []
sub_dirs = os.listdir('./train/')
for sub_dir in sub_dirs:
    file_names.extend(os.listdir(os.path.join('./train', sub_dir)))

abnormals = []
for idx, (image, label) in tqdm(enumerate(train_dataloader)):
    image, label = image.cuda(), label.cuda()
    predict = model(image)
    pred_y = torch.argmax(predict, dim=1)
    max_pred = torch.max(predict)
    
    if pred_y != label or max_pred < 0.6:
        abnormals.append(file_names[idx])
        print(f'{file_names[idx]}, {predict}')

for abnormal in abnormals:
    f.write(abnormal + '\n')
f.close()