import torch



def Getloss(stragety):
    if stragety=="MSE":
        MSE = torch.nn.MSELoss(reduce=True, size_average=True)
        return MSE

