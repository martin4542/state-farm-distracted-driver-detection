# state-farm-distracted-driver-detection

Kaggle State farm distracted driver detection competition
https://www.kaggle.com/c/state-farm-distracted-driver-detection

Pytorch 1.9.0

If you want to test on validation set while training run
> python val_split.py --mode 1
before training

You can train your model by
> python train.py
If you want to change base line model to ResNet use --model 'ResNet'
