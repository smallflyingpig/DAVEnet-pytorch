import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as imagemodels
import torch.utils.model_zoo as model_zoo

class Resnet18(imagemodels.ResNet):
    def __init__(self, embedding_dim=1024, pretrained=False):
        super(Resnet18, self).__init__(imagemodels.resnet.BasicBlock, [2, 2, 2, 2])
        if pretrained:
            self.load_state_dict(model_zoo.load_url(imagemodels.resnet.model_urls['resnet18']))
        self.avgpool = None
        self.fc = None
        self.embedder = nn.Conv2d(512, embedding_dim, kernel_size=1, stride=1, padding=0)
        self.embedding_dim = embedding_dim
        self.pretrained = pretrained

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.embedder(x)
        return x

class Resnet34(imagemodels.ResNet):
    def __init__(self, embedding_dim=1024, pretrained=False):
        super(Resnet34, self).__init__(imagemodels.resnet.BasicBlock, [3, 4, 6, 3])
        if pretrained:
            self.load_state_dict(model_zoo.load_url(imagemodels.resnet.model_urls['resnet34']))
        self.avgpool = None
        self.fc = None
        self.embedder = nn.Conv2d(512, embedding_dim, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.embedder(x)
        return x

class Resnet50(imagemodels.ResNet):
    def __init__(self, embedding_dim=1024, pretrained=False):
        super(Resnet50, self).__init__(imagemodels.resnet.Bottleneck, [3, 4, 6, 3])
        if pretrained:
            self.load_state_dict(model_zoo.load_url(imagemodels.resnet.model_urls['resnet50']))
        self.avgpool = None
        self.fc = None
        self.embedder = nn.Conv2d(2048, embedding_dim, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.embedder(x)
        return x

class VGG16(nn.Module):
    def __init__(self, embedding_dim=1024, pretrained=False):
        super(VGG16, self).__init__()
        self.pretrained = pretrained
        self.seed_model = imagemodels.__dict__['vgg16'](pretrained=pretrained).features
        self.seed_model1 = nn.Sequential(*list(self.seed_model.children())[:-1]) # remove final maxpool
        if pretrained:
            for param in self.seed_model.parameters():
                param.requires_grad_(False)
            for param in self.seed_model1.parameters():
                param.requires_grad_(False)
                
        last_layer_index = len(list(self.seed_model.children()))
        self.seed_model2 = nn.Sequential()
        self.seed_model2.add_module(str(last_layer_index),
            nn.Conv2d(512, embedding_dim, kernel_size=(3,3), stride=(1,1), padding=(1,1)))
        # self.image_model = seed_model
        self.global_layer = nn.Sequential(
            nn.MaxPool2d(2, 2, 0),
            nn.Conv2d(1024, 1024, 3,2,0, bias=True),
            nn.AvgPool2d(3, stride=3)
        )

    def forward(self, x):
        # (512, 14, 14)
        if self.pretrained:
            with torch.autograd.no_grad():
                x = self.seed_model1(x)
        else:
            x = self.seed_model1(x)
        x = self.seed_model2(x.requires_grad_())
        feature = self.global_layer(x).squeeze()
        

        #x = self.image_model(x)
        return x, feature
