import torch
import torch.nn as nn
import torchvision.models as models

# --------------------VGG 16 pretrained model--------------------
class get_model(nn.Module):
    def __init__(self, args):
        super(get_model, self).__init__()
        self.model = args.model
        self.fix_param = args.transfer
        self.pretrained = args.pretrained

        # use vgg16 model as feature extractor
        if self.model == 'VGG16':
            base_model = models.vgg16(pretrained=self.pretrained)
            feature_model = nn.Sequential(*list(base_model.children()))
            feature_model = feature_model[0]
            num_feat = 25088

        # use ResNet101 model as feature extractor
        elif self.model == 'ResNet':
            base_model = models.resnet101(pretrained=self.pretrained)
            feature_model = nn.Sequential(*list(base_model.children()))
            feature_model = feature_model[0:8]
            num_feat = 100352
        
        else:
            return -1

        # define feature extractor
        self.feature_model = feature_model
        
        # define classifier
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(num_feat, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 10),
            nn.Softmax(dim=1)
        )
        
        # if you not want to trian feature extractor
        if self.fix_param:
            for param in self.feature_model.parameters():
                param.requires_grad = False

    def forward(self, x):
        feature = self.feature_model(x)
        flatten = self.flatten(feature)
        output = self.classifier(flatten)
        return output