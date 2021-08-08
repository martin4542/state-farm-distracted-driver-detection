import os
import torch
import shutil
import argparse
import torchvision.datasets as dset
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from model import get_model

parser = argparse.ArgumentParser(description='distracted driver detection test')
parser.add_argument('--model', default='VGG16', help='type of base model')
parser.add_argument('--transfer', default=False, help='set true if you want to fix pretrained model parameter')
parser.add_argument('--pretrained', default=True, help='set true if you want pretrained model')
parser.add_argument('--model_path', required=True, help='pretrained model path')
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
model = get_model(args)
model.load_state_dict(torch.load(args.model_path))
model.cuda()
model.eval()
print(model)

# --------------------model testing & data split--------------------
file_names = os.listdir('./test/test/')
src_path = './test/test'
dst_path = './train'
label = ['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9']

for idx, image in tqdm(enumerate(test_dataloader)):
    image = image[0].cuda()
    predict = model(image)

    pred_max = torch.max(predict)
    pred_label = torch.argmax(predict, dim=1)
    
    if pred_max > 0.95:
        file_name = file_names[idx]
        shutil.copy(os.path.join(src_path, file_name), 
                    os.path.join(dst_path, label[pred_label], file_name))