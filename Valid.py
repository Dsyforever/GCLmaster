import torch
from torch.utils.data import DataLoader


def valid(model, v_dataset, top_k, arg):
    data_loaders = DataLoader(v_dataset, batch_size = arg.batchsize, shuffle = True)
    acc_number = 0
    for id, batch in enumerate(data_loaders):
        images, labels = batch
        images = images.cuda()
        labels = labels.cuda()
        outputs = model(images)
        if categories:= outputs.shape[1] <= top_k:
            print('It\'s meaningless to compute {0:top_k} accuracy on a dataset with {1:categories} \
                categories.'.format(top_k = top_k, categories = categories))
            return 1.0
        else:
            #outputs = outputs.copy()
            for i in range(0, top_k):
                if outputs.shape[0]:
                    idxmax = outputs.argmax(dim = 1)
                    idxeq = idxmax.eq(labels)
                    acc_number += idxeq.sum().item()
                    outputs[torch.arange(0, outputs.shape[0], 1), idxmax] = -1
                    idxeq = (idxeq == False)
                    outputs, labels = outputs[idxeq], labels[idxeq]
    accuracy = acc_number / len(v_dataset)
    return accuracy