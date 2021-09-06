import os
import shutil
import sys
import torch.nn.functional as F
import numpy as np
import torch
import argparse
import torch.optim as optim
import models
import torch.nn as nn
from torchvision import datasets, transforms
import pandas as pd

from dataloader import get_poison_data
from get_validation_dataset import get_valid_set

parser = argparse.ArgumentParser(description='PyTorch fine tune')

parser.add_argument('--dataset', type=str, default='cifar10',
                    help='training dataset (default: cifar100)')
parser.add_argument('--data', type=str, default=None,
                    help='path to dataset')
parser.add_argument('--sparsity-regularization', '-sr', dest='sr', action='store_true',
                    help='train with channel sparsity regularization')
parser.add_argument('--s', type=float, default=0.0001,
                    help='scale sparse rate (default: 0.0001)')
parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--validate-batch-size', type=int, default=256, metavar='N',
                    help='input batch size for testing (default: 256)')
parser.add_argument('--epochs', type=int, default=160, metavar='N',
                    help='number of epochs to train (default: 160)')

parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')

parser.add_argument('--arch', default='vgg', type=str,
                    help='architecture to use')
parser.add_argument('--depth', default=16, type=int,
                    help='depth of the neural network')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--schedule', type=int, nargs='+', default=[70, 100],
                    help='Decrease learning rate at these epochs.')

parser.add_argument('--start-epoch', default=16, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')

parser.add_argument('--resume_arch', default='./O_poisonresult_white/1/pruned_5029_0.5/pruned.pth.tar', type=str,
                    metavar='PATH',
                    help='path to the model architecture(default: none)')

parser.add_argument('--resume_weight', default='./O_poisonresult_white/1/pruned_5029_0.5/115checkpoint.pth.tar',
                    type=str,
                    metavar='PATH',
                    help='path to the model weight(default: none)')
parser.add_argument('--csv_file_path', default='./demo.csv', type=str, metavar='PATH',
                    help='path to save result csv file1')

parser.add_argument('--filter', default='none', type=str, choices=['none', 'lowpass', 'highpass'])
parser.add_argument('--sparsity_gt', default=0, type=float, help='sparsity controller')
# -----------------------------New Args For Our Experiment------------------------------------ #
parser.add_argument('--poison_pre', type=int, default=1)
parser.add_argument('--train_num', default=50000, help='train_num')
parser.add_argument('--test_num', default=10000, help='test_num')
parser.add_argument('--alpha_train', default=0.3, help='alpha_train*poison + (1-alpha)*origin')
parser.add_argument('--alpha_test', default=0.3, help='alpha_test')
parser.add_argument('--beta_train', default=0.1, help='poison_train_num = beta_train*all_train_num')
parser.add_argument('--beta_test', default=0.5, help='poison_test_num')
parser.add_argument('--poison_method', type=int, default=1,
                    help='0~no backdoor, 1~white trigger, 2~random trigger, 3~invisible trigger')
# -----------------------------New Args For Our Experiment------------------------------------ #


args = parser.parse_args()
args.cuda = torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


validate_loader = get_valid_set(args.validate_batch_size)
train_loader, test_loader, origin_testloader = get_poison_data(args)

if os.path.isfile(args.resume_arch):
    print("=> loading checkpoint '{}'".format(args.resume_arch))

    checkpoint = torch.load(args.resume_arch)
    model = models.__dict__[args.arch](dataset=args.dataset, depth=args.depth, cfg=checkpoint['cfg'])
    if args.cuda:
        model.cuda()
else:
    print("=> no checkpoint found at '{}'".format(args.resume_arch))
    sys.exit()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

history_score = np.zeros((args.epochs, 3))


# additional subgradient descent on the sparsity-induced penalty term
def updateBN():
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.weight.grad.data.add_(args.s * torch.sign(m.weight.data))  # L1


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def train(epoch):
    model.train()
    global history_score
    avg_loss = 0.
    train_acc = 0.
    for batch_idx, (data, target) in enumerate(validate_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        # data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        avg_loss += loss.item()
        # pred = output.data.max(1, keepdim=True)[1]
        # train_acc += pred.eq(target.data.view_as(pred)).cpu().sum()
        prec1, prec5 = accuracy(output.data, target.data, topk=(1, 5))
        train_acc += prec1.item()
        loss.backward()
        if args.sr:
            updateBN()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))
    # history_score[epoch][0] = avg_loss / len(validate_loader)
    # history_score[epoch][1] = np.round(train_acc / len(validate_loader), 2)


def test():
    model.eval()
    test_loss = 0
    test_acc = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        # data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.cross_entropy(output, target, size_average=False).item()  # sum up batch loss
        # pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        # correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        prec1, prec5 = accuracy(output.data, target.data, topk=(1, 5))
        test_acc += prec1.item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
        test_loss, test_acc, len(test_loader), test_acc / len(test_loader)))
    return np.round(test_acc / len(test_loader), 2)


def test_origin():
    model.eval()
    test_loss = 0
    test_acc = 0
    for data, target in origin_testloader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        # data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.cross_entropy(output, target, size_average=False).item()  # sum up batch loss
        # pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        # correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        prec1, prec5 = accuracy(output.data, target.data, topk=(1, 5))
        test_acc += prec1.item()

    test_loss /= len(test_loader.dataset)
    print('\nOrigin test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)\n'.format(
        test_loss, test_acc, len(test_loader), test_acc / len(test_loader)))
    return np.round(test_acc / len(test_loader), 2)


def save_checkpoint(state, is_best, filepath):
    torch.save(state, os.path.join(filepath, 'checkpoint.pth.tar'))
    if is_best:
        shutil.copyfile(os.path.join(filepath, 'checkpoint.pth.tar'), os.path.join(filepath, 'model_best.pth.tar'))


if os.path.isfile(args.resume_weight):
    print("=> loading checkpoint '{}'".format(args.resume_weight))

    checkpoint = torch.load(args.resume_weight)

    args.start_epoch = checkpoint['epoch']
    best_prec1 = checkpoint['best_prec1']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}"
            .format(args.resume_weight, checkpoint['epoch'], best_prec1))
else:
    print("=> no checkpoint found at '{}'".format(args.resume_weight))
    sys.exit()

for param_group in optimizer.param_groups:
    if args.start_epoch < args.schedule[0]:
        param_group['lr'] = 0.1
    elif args.start_epoch > args.schedule[0] and args.start_epoch < args.schedule[1]:
        param_group['lr'] = 0.01
    elif args.start_epoch > args.schedule[1]:
        param_group['lr'] = 0.001
save_list = []
save_name = ['epoch', 'ASR', 'CDA']

prec = test()
if args.poison_pre == 1:
    origin_prec = test_origin()
save_list.append([0, prec, origin_prec])

for epoch in range(args.start_epoch, args.start_epoch + 30):
    if epoch in args.schedule:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1
    train(epoch)
    prec = test()
    if args.poison_pre == 1:
        origin_prec = test_origin()
    save_list.append([epoch - args.start_epoch + 1, prec, origin_prec])

save_pd = pd.DataFrame(columns=save_name, data=save_list)
save_pd.to_csv(args.csv_file_path)
print('save data to' + args.csv_file_path)