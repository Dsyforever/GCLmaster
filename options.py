import argparse
import sys


def print_options(parser, opt, file=sys.stdout):
    """Print and save options
    It will print both current options and default values(if different).
    It will save options into a text file / [checkpoints_dir] / opt.txt
    """
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(opt).items()):
        comment = ''
        default = parser.get_default(k)
        if v != default:
            comment = '\t[default: %s]' % str(default)
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    print(message, file=file)


def parse_args():
    parser = argparse.ArgumentParser(description='CL-master')
    parser.add_argument('--task', type=str, default="CIFAR10")  # 任务
    parser.add_argument('--stragety', type=str, default="CrossEntropy")  # 策略
    parser.add_argument('--data_dir', type=str, default="./data")  # 数据集路径
    parser.add_argument('--batchsize', type=int, default=356)  # BS
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--backbone', type=str, default="resnet18")
    parser.add_argument('--CLoptimizer', type=str, default="SGD")
    parser.add_argument('--Q', type=float, default=0.05)
    parser.add_argument('--Epoch', type=int, default=100)
    parser.add_argument('--pretrain_param', type=bool, default=False)
    parser.add_argument('--topology', type=int, default=0)
    parser.add_argument('--log_root', type=str, default='none') # 日志的保存地址。如果为none，就不保存日志。不需要加.txt
    parser.add_argument('--is_picture', type=int, default=0) # 是否保存图片，如果保存，图片将会和日志存储在同一个文件夹，并且和日志名字相同
    args = parser.parse_args()
    return args, parser
