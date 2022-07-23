import torch
from Dataset import Getnumclass
#Valid dataset



def valid(model,v_dataset,arg):
    data_loaders = torch.utils.data.DataLoader(v_dataset, batch_size=arg.batchsize, shuffle=True)
    acc_number_total=0
    for id, batch in enumerate(data_loaders):
        images, labels = batch
        images = images.cuda()
        labels = labels.cuda()
        outs = model(images)
        acc_number=outs.argmax(dim=1).eq(labels).sum().item()
        acc_number_total+=acc_number
    accuracy=acc_number_total/len(v_dataset)
    return accuracy