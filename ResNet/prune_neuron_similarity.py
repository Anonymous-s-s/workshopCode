import argparse
import numpy as np
import os

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import cv2
import PIL.Image

# from models import *
import models

# Prune settings
parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR prune')
parser.add_argument('--dataset', type=str, default='cifar100',
                    help='training dataset (default: cifar10)')
parser.add_argument('--val-batch-size', type=int, default=256, metavar='N',
                    help='input batch size for validatin (default: 256)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--depth', type=int, default=18,
                    help='depth of the vgg')
parser.add_argument('--arch', default='resnet18_cifar', type=str,
                    help='architecture to use')
# parser.add_argument('--model', default='', type=str, metavar='PATH',
#                     help='path to the model (default: none)')
parser.add_argument('--save', default='./cleanresult/4/EB-30-35.pth.tar', type=str, metavar='PATH',
                    help='path to save pruned model (default: none)')
parser.add_argument('--save_1', default='./poisonresult_2/5/EB-50-35.pth.tar', type=str, metavar='PATH',
                    help='path to save pruned model (default: none)')

# parser.add_argument('--save_2', default='./poisonresult_2/3/EB-30-27.pth.tar', type=str, metavar='PATH',
#                     help='path to save pruned model (default: none)')
# parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='manual start epoch number')
# parser.add_argument('--end_epoch', default=160, type=int, metavar='N', help='manual end epoch number')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

print('-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-')
print('Experiment Starting... Check critical information below carefully!')
print('Training Phase: Calculate Difference of Two Masks;')
print('Dataset:{};'.format(args.dataset))
# print('Dataset:{};\tStart Epoch:{};\tEnd Epoch:{};'.format(args.dataset, args.start_epoch, args.end_epoch))  #
print('Network Architecture:{};\tDepth:{};'.format(args.arch, args.depth))  #
print('First Mask Path:{};'.format(args.save))
print('Second Mask Path:{};'.format(args.save_1))
print('-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-')

setting_perc = 0.7

if not os.path.exists(args.save):
    os.makedirs(args.save)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.__dict__[args.arch](dataset=args.dataset, depth=args.depth)
model_bd = models.__dict__[args.arch](dataset=args.dataset, depth=args.depth)

if args.cuda:
    model.cuda()
    model_bd.cuda()


def pruning(model, percent):
    total = 0
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            total += m.weight.data.shape[0]

    bn = torch.zeros(total)
    index = 0
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            size = m.weight.data.shape[0]
            bn[index:(index + size)] = m.weight.data.abs().clone()
            index += size

    y, i = torch.sort(bn)
    thre_index = int(total * percent)
    thre = y[thre_index]

    mask2 = bn.gt(thre).float().view(-1)
    return mask2


# get clean EB
print('==> resumeing from {} ... '.format(args.save))
checkpoint = torch.load(args.save)
best_epoch = checkpoint['epoch']
print('EarlyBird Emerging Epoch: ', best_epoch)
model.load_state_dict(checkpoint['state_dict'])

# get backdoor EB and mask
print('==> resumeing from {} ... '.format(args.save_1))
checkpoint_bd = torch.load(args.save_1)
best_epoch_bd = checkpoint_bd['epoch']
print('EarlyBird Emerging Epoch: ', best_epoch_bd)
model_bd.load_state_dict(checkpoint_bd['state_dict'])
best_mask_bd = pruning(model_bd, setting_perc)

X = []
Y = []

for percent_set in np.arange(setting_perc, 0.98, 0.05):  # [0.3, 0.35, 0.4, ... , 1]:
    X.append(percent_set)
    print("\nclean prune precent:", percent_set)
    best_mask = pruning(model, percent_set)  # get clean mask

    in_num = 0
    for i in range(best_mask.size(0)):
        if best_mask[i] == 1 and best_mask_bd[i] == 1:
            in_num += 1
    Y.append(in_num / int(torch.sum(best_mask)))
    print("both exist percent:", in_num / int(torch.sum(best_mask)))
print(X)
print(Y)

# ----------------------
# mask_length = len(best_mask.data)
# best_mask_square = best_mask.clone().numpy()
# best_mask_square = best_mask_square.reshape(int(mask_length ** 0.5), -1)
# best_mask_bd_square = best_mask_bd.clone().numpy()
# best_mask_bd_square = best_mask_bd_square.reshape(int(mask_length ** 0.5), -1)
# combine_square = best_mask_square * 0.5 + best_mask_bd_square * 0.5
#
# fig = plt.figure()
# fig.suptitle("Masks Comparison", fontsize=16)
# left = plt.subplot(1, 3, 1)
# left.set_title(args.save.split('.')[1].split('/')[2])
# left.imshow(best_mask_square, cmap=plt.cm.gray)
#
# right = plt.subplot(1, 3, 2)
# right.set_title(args.save_1.split('.')[1].split('/')[2])
# right.imshow(best_mask_bd_square, cmap=plt.cm.gray)
#
# com = plt.subplot(1, 3, 3)
# com.set_title('Combine Masks')
# com.imshow(combine_square, cmap=plt.cm.gray)
#
# plt.tight_layout()
# plt.show()
# plt.pause(0.001)
