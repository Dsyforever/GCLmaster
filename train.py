import argparse
from Model import Getmodel
from Dataset import Getdataset,Getnumclass
from Optimizers import Getoptim
from stragety import  Getloss
from Valid import valid
import torch
from options import *

if __name__ == "__main__":
    # get arguments
    args = parse_args()
    num_class = Getnumclass(args.task)
    train_dataset,test_dataset= Getdataset(args.task,args.data_dir)
    model= Getmodel(args.task,args.backbone,num_class,args.stragety,args.pretrain_param)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True)
    Hparam = [{"params": model.parameters(), "lr": args.lr}]
    optimizer = Getoptim(args.CLoptimizer,Hparam)
    model=model.cuda()
    lossf=Getloss(args.stragety)

    # use to record the result
    result = {'epoch':[], 'train': {'acc': []}, 'test': {'acc': []}}
    # begin to train
    for epoch in tqdm(args.Epoch):
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
        test_accurary = valid(model, test_dataset, args)

        result['train']['acc'].append(train_accurary)
        result['test']['acc'].append(test_accurary)
        result['epoch'].append(epoch)

        print("{epoch_index} epoch's train_accurary is {trainacc},train_loss is {losst}, test_accurary is {testacc}.".format(epoch_index=epoch,trainacc=train_accurary,losst=total_loss ,testacc=test_accurary))

        #save checkpoint
        if epoch%20==0:
            state = {
                'epoch': epoch ,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),  # 保存优化器参数
                    }
            torch.save(state, './checkpoint/top/checkpoint_epoch{index}.pth.tar'.format(index=epoch))
