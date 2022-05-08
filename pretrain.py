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
import shutil
import math
import os
import time
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.utils.data
from cl.loss import ConsistentContinuousLoss
from cl.gpuInfo import check_gpu_mem_used_rate
from dataset.VideoDataset import VideoDataset

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith('__') and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch Video Pretrain')
parser.add_argument('--seed', default=None, type=int, help='seed for initializing training.')
parser.add_argument('dataset', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50', choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of dataset loading workers (default: 32)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=4, type=int, metavar='N',
                    help='mini-batch size (default: 256), this is the total batch size of all GPUS'
                         'on the current node when using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning_rate', default=0.03, type=float, metavar='LR', dest='lr',
                    help='initial learning rate')
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--cos', action='store_true',
                    help='use cosine lr schedule')
parser.add_argument('--wd', '--weight_decay', default=1e-4, type=float, metavar='W', dest='weight_decay',
                    help='weight decay (default: 1e-4)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none).')
parser.add_argument('-p', '--print-freq', default=10, type=int, metavar='N',
                    help='print frequency (default: 10)')

# Distributed Config
parser.add_argument('--world_size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist_url', default='env://127.0.0.1:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist_backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch N processes per node, which has N GPUs.'
                         'This is the fastest way to use PyTorch for '
                         'either single node or multi node dataset parallel training')

# Model config
parser.add_argument('--num_frames', default=50, type=int,
                    help='number of frames extracted from each video')


def main():
    # Parameters and Distributed config
    args = parser.parse_args()
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '23456'

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed traning.'
                      'This will turn on the CUDNN deterministic setting,'
                      'which can slow down your training considerably!'
                      'You may see unexpected behavior when restarting from checkpoints')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely disable dataset parallel.')

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
        # Simply call main_worker function
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
        # dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
        #                         world_size=args.world_size, rank=args.rank)
        dist.init_process_group(backend=args.dist_backend, init_method="env://127.0.0.1:23456",
                                world_size=args.world_size, rank=args.rank)

    # Create model - ResNet50 or ViT
    print("=> create model '{}'".format(args.arch))
    model = models.__dict__[args.arch]()
    # print(model)

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
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], broadcast_buffers=False)
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch size to
            # all available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model, broadcast_buffers=False)
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
    criterion = ConsistentContinuousLoss().cuda(args.gpu)
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

    # Load dataset
    # data_loader = load_data(train_dir=args.dataset, batch_size=args.batch_size, shuffle=True)
    # dataset = load_data(train_dir=args.data, num_frames=100)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    augmentation = transforms.Compose([
        transforms.CenterCrop(512),
        transforms.Resize((224, 224)),
        transforms.RandomGrayscale(p=0.2),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
        # transforms.ToTensor(),
        normalize
    ])
    dataset = VideoDataset(data_dir=args.dataset, transforms=augmentation, num_frames=args.num_frames)
    data_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, sampler=data_sampler, collate_fn=VideoDataset.collate_fn, drop_last=True)

    # Training
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args)

        # Train for one epoch
        train_one_epoch(data_loader, model, criterion, optimizer, epoch, args)

        if not args.multiprocessing_distributed or \
                (args.multiprocessing_distributed and args.rank * ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }, is_best=False, filename='checkpoint_{:04d}.pth.tar'.format(epoch))


def train_one_epoch(data_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    consistent_loss = AverageMeter('Loss@consistent', ':.4e')
    continuous_loss = AverageMeter('Loss@Continuous', ':.4e')
    progress = ProgressMeter(
        len(data_loader),
        [batch_time, data_time, losses, consistent_loss, continuous_loss],
        prefix='Epoch: [{}]'.format(epoch)
    )

    # Switch to train mode
    model.train()

    end = time.time()
    for i, frames in enumerate(data_loader):
        # Measure dataset loading time
        data_time.update(time.time() - end, 1)

        check_gpu_mem_used_rate(f"Batch_{i}_start")

        if args.gpu is not None:
            frames = frames.cuda(args.gpu, non_blocking=True)

        # Compute output
        # The input dimension of model(resnet or vit) is (N, channels, height, width),
        # but for our pretraining, the input dimension is (N, frames, channels, height, width),
        # so we need to adjust the input dimension to fit the model.
        cnt_frames = frames.shape[1]
        outputs = []
        for j in range(cnt_frames):
            frame = frames[:, j, :, :, :]  # (N, channels, height, width)
            output = model(frame)  # (N, dim)
            outputs.append(output)
            # check_gpu_mem_used_rate(f"Batch_{i}_{j}_forward")
        outputs = torch.stack(outputs, dim=1)  # (N, frames, dim)

        check_gpu_mem_used_rate(f"Batch_{i}_forward")

        cons_loss, cont_loss = criterion(outputs)
        loss = cons_loss + cont_loss

        print(f"loss = {loss}")

        check_gpu_mem_used_rate(f"Batch_{i}_loss")

        # Record loss
        losses.update(loss, frames.size(0))
        consistent_loss.update(cons_loss, frames.size(0))
        continuous_loss.update(cont_loss, frames.size(0))

        # Compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        check_gpu_mem_used_rate(f"Batch_{i}_backward")

        # Measure elapsed time
        batch_time.update(time.time() - end, 1)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """ Compute and store the average and current value """
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.value = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.value = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {value' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """ Decay the learning rate based on schedule """
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 *(1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()
    # frames = torch.rand((3, 10, 3, 224, 224))
    # model = models.resnet50(pretrained=True)
    # cnt_frames = frames.shape[1]
    # outputs = []
    # for j in range(cnt_frames):
    #     frame = frames[:, j, :, :, :]  # (N, channels, height, width)
    #     output = model(frame)  # (N, dim)
    #     outputs.append(output)
    # outputs = torch.stack(outputs, dim=1)
    # print(f'outputs.shape = {outputs.shape}')
