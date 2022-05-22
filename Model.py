import torch
import torchvision


class Softmaxnet(torch.nn.Module):
    def __init__(self, in_feature, out_feature):
        super().__init__()
        self.linears = torch.nn.Linear(in_features=in_feature, out_features=out_feature)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.linears(x)
        x = self.softmax(x)
        return x

def Getmodel(task,backbone,n_classes):
    if backbone=="resnet50":
        model= torchvision.models.resnet50(pretrained=False)

    if task=="CIFAR10":
        model.fc = Softmaxnet(2048, n_classes)
    return model