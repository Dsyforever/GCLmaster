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
    parser.add_argument('--stragety', type=str, default="CrossEntropy")#策略
    parser.add_argument('--data_dir', type=str,default="./data")#数据集路径
    parser.add_argument('--batchsize', type=int, default=356)#BS
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--backbone', type=str, default="resnet50")
    parser.add_argument('--CLoptimizer', type=str, default="SGD")
    parser.add_argument('--Q', type=float, default=0.05)
    parser.add_argument('--Epoch', type=int, default=100)
    parser.add_argument('--pretrain_param', type=bool, default=False)
    parser.add_argument('--topology', type=int, default=0)
    args = parser.parse_args()


    num_class = Getnumclass(args.task)
    train_dataset,test_dataset= Getdataset(args.task,args.data_dir)
    model= Getmodel(args.task,args.backbone,num_class,args.stragety,args.pretrain_param)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True)
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
            
            #topology loss by metic
            topology_loss=0
            lossft= torch.nn.MSELoss(reduce=True)
            if args.topology!=0:
                for i in range(args.topology): 
                    dxi=args.Q*2*(torch.rand(images.shape)-0.5).to(images.device)+images
                    topology_loss+=lossft(outs,model(dxi))
                topology_loss=topology_loss/args.topology
                
                    
            loss = lossf(outs, labels_cal)+topology_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_accurary = valid(model, train_dataset, args)
        test_accurary=valid(model,test_dataset,args)
        print("{epoch_index} epoch's train_accurary is {trainacc},train_loss is {losst}, test_accurary is {testacc}.".format(epoch_index=epoch,trainacc=train_accurary,losst=total_loss ,testacc=test_accurary))
        #save checkpoint
        if epoch%20==0:
            state = {
                'epoch': epoch ,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),  # 保存优化器参数
                    }
            torch.save(state, './checkpoint/top/checkpoint_epoch{index}.pth.tar'.format(index=epoch))
