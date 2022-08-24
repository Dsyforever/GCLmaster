import torch
from torch.utils.data import DataLoader


def valid(model, v_dataset, arg, device):
    data_loaders = DataLoader(v_dataset, batch_size = arg.batchsize, shuffle = True)
    acc_number = 0
    for id, batch in enumerate(data_loaders):
        images, labels = batch
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        if categories:= outputs.shape[1] <= arg.top_k:
            print('It\'s meaningless to compute top{0:top_k} accuracy on a dataset with {1:categories} \
                categories.'.format(top_k = arg.top_k, categories = categories))
            return 1.0
        else:
            for i in range(0, arg.top_k):
                if outputs.shape[0]:
                    idxmax = outputs.argmax(dim = 1)
                    idxeq = idxmax.eq(labels)
                    acc_number += idxeq.sum().item()
                    outputs[torch.arange(0, outputs.shape[0], 1), idxmax] = -1
                    idxeq = (idxeq == False)
                    outputs, labels = outputs[idxeq], labels[idxeq]
    accuracy = acc_number / len(v_dataset)
    return accuracy