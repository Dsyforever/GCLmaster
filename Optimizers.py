import torch

def Getoptim(CLoptimizer,Hparam):
    if CLoptimizer=="SGD":
        return torch.optim.SGD(Hparam)