import torch
import timm
#预训练模型
model = timm.create_model('wide_resnet50_2', pretrained=True,num_classes=0)
model.eval()
#随机初始化模型
model1=timm.create_model('wide_resnet50_2',pretrained=False,num_classes=0)
model1.eval()

#model.head=torch.nn.Linear()
print(model)