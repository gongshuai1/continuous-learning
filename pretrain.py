# !/usr/bin/env python
# =====================================================================================
#
# @Date: 2022-02-28 16:22
# @Author: gongshuai
# @File: pretrain.py
# @IDE: PyCharm
# @Func: pretrain ResNet/ViT using video dataset from scratch
#
# =====================================================================================
import argparse
import random
import warnings
import builtins
import os
import torch
import torchvision.models as models
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.distributed as dist
from cl.loss import ConsistentContinuousLoss

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith('__') and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch Video Pretrain')
parser.add_argument('--seed', default=None, type=int, help='seed for initializing training.')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50', choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=256, type=int, metavar='N',
                    help='mini-batch size (default: 256), this is the total batch size of all GPUS'
                         'on the current node when using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning_rate', default=0.03, type=float, metavar='LR', dest='lr',
                    help='initial learning rate')
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--wd', '--weight_decay', default=1e-4, type=float, metavar='W', dest='weight_decay',
                    help='weight decay (default: 1e-4)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none).')

# Distributed Config
parser.add_argument('--world_size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist_url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist_backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch N processes per node, which has N GPUs.'
                         'This is the fastest way to use PyTorch for '
                         'either single node or multi node data parallel training')


def main():
    # Parameters and Distributed config
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed traning.'
                      'This will turn on the CUDNN deterministic setting,'
                      'which can slow down your training considerably!'
                      'You may see unexpected behavior when restarting from checkpoints')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely disable data parallel.')

    if args.dist_url == 'env://' and args.world_size == -1:
        args.world_size = int(os.environ['WORLD_SIZE'])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()  # the number of gpus in per node
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call mian_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    # Suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        print('Use GPU: {} for training'.format(args.gpu))

    if args.distributed:
        if args.dist_url == 'env://' and args.rank == -1:
            args.rank = int(os.environ['RANK'])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    # Create model - ResNet50 or ViT
    print("=> create model '{}'".format(args.arch))
    model = models.__dict__[args.arch]
    print(model)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributeDataParallel will use all available devices
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using single GPU per process and per DistributedDataParallel,
            # we need to divide batch size ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch size to
            # all available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # Comment out the following line for debugging
        raise NotImplementedError('Only DistributedDaraParallel is supported')
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError('Only DistributedDataParallel is supported')

    # Define loss function and optimizer
    loss = ConsistentContinuousLoss().cuda(args.gpu)
    optimizer = torch.optim.SGD(model.parameters(), args.lr, weight_decay=args.weight_decay)

    # Optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Load data


def train_one_epoch():
    pass


if __name__ == '__main__':
    main()
