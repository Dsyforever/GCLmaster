import torchvision
from torchvision import transforms
def Getdataset(task,dir):
    DATASET = None
    test_DATASET = None
    val_DATASET = None
    myTransforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    if task == "CIFAR10":
        DATASET = torchvision.datasets.CIFAR10(root=dir, download=True, transform=myTransforms, train=True)
        test_DATASET = torchvision.datasets.CIFAR10(root=dir, download=True, transform=myTransforms, train=False)

    if task == "CIFAR100":
        DATASET = torchvision.datasets.CIFAR100(root=dir, download=False, transform=myTransforms, train=True)
        test_DATASET = torchvision.datasets.CIFAR100(root=dir, download=False, transform=myTransforms, train=False)

    if task == "CIFAR10_noresize":
        DATASET = torchvision.datasets.CIFAR10(root=dir, download=False, transform=torchvision.transforms.ToTensor(),
                                               train=True)
        test_DATASET = torchvision.datasets.CIFAR10(root=dir, download=False,
                                                    transform=torchvision.transforms.ToTensor(), train=False)

    if task == "DTD":
        DATASET = torchvision.datasets.DTD(root=dir, download=False, transform=myTransforms, split='train')
        val_DATASET = torchvision.datasets.DTD(root=dir, download=False, transform=myTransforms, split='val')
        test_DATASET = torchvision.datasets.DTD(root=dir, download=False, transform=myTransforms, split='test')

    if task == "Flowers102":
        DATASET = torchvision.datasets.Flowers102(root=dir, download=False, transform=myTransforms, split='train')
        val_DATASET = torchvision.datasets.Flowers102(root=dir, download=False, transform=myTransforms, split='val')
        test_DATASET = torchvision.datasets.Flowers102(root=dir, download=False, transform=myTransforms, split='test')

    # if task == "SUN397":
    #     DATASET = torchvision.datasets.SUN397(root=dir, download=True, transform=myTransforms)

    # if task == "Caltech101":
    #     DATASET = torchvision.datasets.Caltech101(root=dir, download=False, transform=myTransforms)

    if task == "Food101":
        DATASET = torchvision.datasets.Food101(root=dir, download=False, transform=myTransforms, split='train')
        test_DATASET = torchvision.datasets.Food101(root=dir, download=False, transform=myTransforms, split='test')

    return DATASET, test_DATASET, val_DATASET
    

def Getnumclass(task):
    if task =="CIFAR10" or task =="CIFAR10_noresize":
        numclass=10
    elif task == "SUN397":
        numclass=397
    elif task == "DTD":
        numclass=47
    elif task == 'Caltech101' or task == 'Aircraft' or task == "Flowers102":  # 数据集介绍那里貌似说除了101个类还有背景这个类
        numclass=102
    elif task == 'CIFAR100':
        numclass=100
    elif task == 'Food101':
        numclass=101
    elif task == 'Cars':
        numclass=196
    elif task == 'Pets':
        numclass=37
    elif task == 'VOC2007':
        numclass=20
    elif task == 'Birdsnap':
        numclass=500
    return numclass