import torch
# from  torch import
import torchvision
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
    if backbone=="resnet18":
        if pretrain_param==True:
            model= torchvision.models.resnet18(pretrained=True)
        else:
            model= torchvision.models.resnet18(pretrained=False)
    if backbone[:3]=='vit':
        model=define_Vit(backbone,n_classes,pretrain_param)

    
    if (task=="CIFAR10" or task=="CIFAR10_noresize")  and stragety != "CrossEntropy":
        model.fc = Softmaxnet(2048, n_classes)
    elif backbone=="resnet50":
        model.fc = torch.nn.Linear(2048, n_classes)
    elif backbone=="resnet18":
        model.fc = torch.nn.Linear(512, n_classes)
        
    return model



