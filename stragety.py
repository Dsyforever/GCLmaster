import torch



def Getloss(stragety):
    if stragety=="MSE":
        MSE = torch.nn.MSELoss(reduce=True, size_average=True)
        return MSE
    if stragety=="CrossEntropy":
        Cross=torch.nn.CrossEntropyLoss()
        return Cross

