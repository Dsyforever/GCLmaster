import argparse
from Model import Getmodel
from Dataset import Getdataset,Getnumclass
from Optimizers import Getoptim
from stragety import  Getloss
from Valid import valid
import torch
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CL-master')
    parser.add_argument('--task', type=str,default="CIFAR10")#任务
    parser.add_argument('--stragety', type=str, default="MSE")#策略
    parser.add_argument('--data_dir', type=str,default="./data")#数据集路径
    parser.add_argument('--batchsize', type=int, default=356)#BS
    parser.add_argument('--lr', type=float, default=1.2e-3)
    parser.add_argument('--backbone', type=str, default="resnet50")
    parser.add_argument('--CLoptimizer', type=str, default="SGD")
    parser.add_argument('--Q', type=float, default=0.05)
    parser.add_argument('--Epoch', type=int, default=100)
    args = parser.parse_args()

    num_class = Getnumclass(args.task)
    dataset= Getdataset(args.task,args.data_dir)
    model= Getmodel(args.task,args.backbone,num_class)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batchsize, shuffle=True)
    Hparam = [{"params": model.parameters(), "lr": args.lr}]
    optimizer = Getoptim(args.CLoptimizer,Hparam)
    model=model.cuda()
    lossf=Getloss(args.stragety)



    for epoch in range(args.Epoch):
        total_loss = 0
        train_correct = 0
        test_correct = 0
        for id, batch in enumerate(train_loader):
            images, labels = batch
            images = images.cuda()
            labels = labels.cuda()
            labels_cal = torch.nn.functional.one_hot(labels,num_class).type(torch.float32).cuda()
            outs = model(images)
            loss = lossf(outs, labels_cal)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()