import torchvision
from torchvision import transforms
def Getdataset(task,dir):
    myTransforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    if task =="CIFAR10":
        DATASET = torchvision.datasets.CIFAR10(root = dir,download=False,transform = myTransforms ,train=True)
        test_DATASET=torchvision.datasets.CIFAR10(root = dir,download=False,transform = myTransforms ,train=False)
    if task =="CIFAR10_noresize":
        DATASET = torchvision.datasets.CIFAR10(root = dir,download=False,transform =  torchvision.transforms.ToTensor() ,train=True)
        test_DATASET=torchvision.datasets.CIFAR10(root = dir,download=False,transform =  torchvision.transforms.ToTensor() ,train=False)
    return DATASET,test_DATASET
    

def Getnumclass(task):
    if task =="CIFAR10" or task =="CIFAR10_noresize":
        numclass=10
    return numclass