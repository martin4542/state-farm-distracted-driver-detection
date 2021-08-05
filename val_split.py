import os
import shutil
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=int, required=True)
args = parser.parse_args()

if not os.path.exists('./val'):
    os.mkdir('./val')

for i in range(10):
    path = os.path.join('val', 'c' + str(i))
    if not os.path.exists(path):
        os.mkdir(path)

if args.mode == 1:
    for i in range(10):
        src_path = os.path.join('train', 'c' + str(i))
        dst_path = os.path.join('val', 'c' + str(i))
        file_names = os.listdir(src_path)
        
        random.shuffle(file_names)

        val_files = file_names[int(len(file_names)*0.8):]
        for file_name in val_files:
            shutil.move(os.path.join(src_path, file_name), os.path.join(dst_path, file_name))

if args.mode == 2:
    for i in range(10):
        src_path = os.path.join('val', 'c' + str(i))
        dst_path = os.path.join('train', 'c' + str(i))
        file_names = os.listdir(src_path)

        for file_name in file_names:
            shutil.move(os.path.join(src_path, file_name), os.path.join(dst_path, file_name))