from options import *
import os
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import torch


def print_root(args):
    # get the time
    t = time.strftime('%m月%d日-%H时%M分', time.localtime(time.time()))
    os.makedirs(args.log_root + '_(' + t + ')', exist_ok=True)
    # create dir for model save
    os.makedirs(args.log_root + '_(' + t + ')/checkpoint', exist_ok=True)
    if args.log_root != 'none':
        f = open(args.log_root + '_(' + t + ')/' + args.log_root + '_(' + t + ')' + '.txt', 'a')
    else:
        f = sys.stdout
    return f, t


def picture(args, result, time):
    plt.figure()
    train_loss = plt.plot(result['epoch'], result['train']['acc'], color='red', linestyle='-.')
    test_loss = plt.plot(result['epoch'], result['test']['acc'], color='blue', linestyle='--')
    if result['val']['acc'][-1] != 0:
        val_loss = plt.plot(result['epoch'], result['val']['acc'], color='green', linestyle='-')
    plt.title('acc vs. epoch(train:red, test:blue)')

    plt.savefig(args.log_root + '_(' + time + ')/' + args.log_root + '_(' + time + ')' + ".png")


def model_save(args,model,optimizer, epoch, loss,time):
    torch.save({
        'epoch': epoch,  # 保存是第几个epoch的模型
        'model_state_dict': model.state_dict(),  # 保存模型的参数而非模型的全部
        'optimizer_state_dict': optimizer.state_dict(),  # 保存优化器参数
        'task': args.task,  # 保存任务
        'loss': loss
    }, args.log_root + '_(' + time + ')/checkpoint'+ args.log_root + '_(' + time + ')' +".pkl")

    # import os
    # from tqdm import tqdm
    # import time
    # import sys
    #
    #
    # class Logger(object):
    #     def __init__(self, filename="lsc.txt"):
    #         self.terminal = sys.stdout
    #         self.log = open(filename, "w")
    #
    #     def write(self, message):
    #         # self.terminal.write(message)
    #         self.log.write(message)
    #
    #     def flush(self):
    #         pass

    # sys.stdout = Logger('a.txt')
    # for i in range(10):
    #     print(i)
    # with open('lsc.txt', 'w') as f:
    #     f.write('lsc test')
    #     for epoch in tqdm(range(10), file=f):
    #         time.sleep(1)
    #         print('\n[{:s}]\t\t{:s}: {:.4f}'.format('Train', 'Acc', epoch), file=f)
    #         print('[{:s}]\t\t{:s}: {:.4f}\n'.format('Test', 'Acc', epoch), file=f)
