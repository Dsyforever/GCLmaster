import torchvision

def Getdataset(task,dir):
    if task =="CIFAR10":
        DATASET = torchvision.datasets.CIFAR10(root = dir,download=False,transform = torchvision.transforms.ToTensor(),train=True)
    return DATASET

def Getnumclass(task):
    if task =="CIFAR10":
        numclass=10
    return numclass