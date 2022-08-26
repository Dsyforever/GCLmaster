import torch
# from  torch import
import torchvision
import timm
from Vits import define_Vit

class Softmaxnet(torch.nn.Module):
    def __init__(self, in_feature, out_feature):
        super().__init__()
        self.linears = torch.nn.Linear(in_features=in_feature, out_features=out_feature)
        self.softmax = torch.nn.Softmax(dim=1)
    def forward(self, x):
        x = self.linears(x)
        x = self.softmax(x)
        return x

def Getmodel(task,backbone,n_classes,stragety,pretrain_param):
    if backbone=="resnet50":
        if pretrain_param==True:
            model= torchvision.models.resnet50(pretrained=True)
        else:
            model= torchvision.models.resnet50(pretrained=False)

    if backbone=="resnet50_2":
        if pretrain_param==True:
            model = timm.create_model('wide_resnet50_2', pretrained=True, num_classes=0)
        else:
            model = timm.create_model('wide_resnet50_2', pretrained=False, num_classes=0)

    if backbone=="resnet101":
        if pretrain_param==True:
            model= torchvision.models.resnet101(pretrained=True)
        else:
            model= torchvision.models.resnet101(pretrained=False)

    if backbone=="resnet101_2":
        if pretrain_param == True:
            model = timm.create_model('wide_resnet101_2', pretrained=True, num_classes=0)
        else:
            model = timm.create_model('wide_resnet101_2', pretrained=False, num_classes=0)

    if backbone[:3]=='vit':
        model=define_Vit(backbone,n_classes,pretrain_param)

    if (task=="CIFAR10" or task=="CIFAR10_noresize")  and stragety != "CrossEntropy":
        model.fc = Softmaxnet(2048, n_classes)
    elif backbone[:8]=="resnet50":
        model.fc = torch.nn.Linear(2048, n_classes)
    elif backbone[:9]=="resnet101":
        model.fc = torch.nn.Linear(2048, n_classes)
        
    return model



