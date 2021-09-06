import argparse
import numpy as np
import os

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import cv2
import PIL.Image

# from models import *
import models

# Prune settings
parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR prune')
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='training dataset (default: cifar10)')
parser.add_argument('--val-batch-size', type=int, default=256, metavar='N',
                    help='input batch size for validatin (default: 256)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--depth', type=int, default=16,
                    help='depth of the vgg')
parser.add_argument('--arch', default='vgg_16', type=str,
                    help='architecture to use')
# parser.add_argument('--model', default='', type=str, metavar='PATH',
#                     help='path to the model (default: none)')
parser.add_argument('--save', default='./cleanresult/1/EB-30-29.pth.tar', type=str, metavar='PATH',
                    help='path to save pruned model (default: none)')
parser.add_argument('--save_1', default='./poisonresult_2/2/EB-30-28.pth.tar', type=str, metavar='PATH',
                    help='path to save pruned model (default: none)')

parser.add_argument('--save_2', default='./poisonresult_2/3/EB-30-27.pth.tar', type=str, metavar='PATH',
                    help='path to save pruned model (default: none)')
# parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='manual start epoch number')
# parser.add_argument('--end_epoch', default=160, type=int, metavar='N', help='manual end epoch number')

# # quantized parameters
# parser.add_argument('--bits_A', default=8, type=int, help='input quantization bits')
# parser.add_argument('--bits_W', default=8, type=int, help='weight quantization bits')
# parser.add_argument('--bits_G', default=8, type=int, help='gradient quantization bits')
# parser.add_argument('--bits_E', default=8, type=int, help='error quantization bits')
# parser.add_argument('--bits_R', default=16, type=int, help='rand number quantization bits')
#
# # multi-gpus
# parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

validation_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('./data.cifar10', train=True, download=True,
                     transform=transforms.Compose([
                         transforms.Pad(4),
                         transforms.RandomCrop(32),
                         transforms.RandomHorizontalFlip(),
                         transforms.ToTensor(),
                         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                     ])),
    batch_size=args.val_batch_size, shuffle=True, **kwargs)

print(
    '-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=')
print('Experiment Starting... Check critical information below carefully!')
print('Training Phase: Calculate Difference of Two Masks;')
print('Dataset:{};'.format(args.dataset))
# print('Dataset:{};\tStart Epoch:{};\tEnd Epoch:{};'.format(args.dataset, args.start_epoch, args.end_epoch))  #
print('Network Architecture:{};\tDepth:{};'.format(args.arch, args.depth))  #
print('First Mask Path:{};'.format(args.save))
print('Second Mask Path:{};'.format(args.save_1))
print(
    '-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=')

setting_perc = 0.3

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
    # print('Pruning threshold: {}'.format(thre))

    # mask = torch.zeros(total)
    # index = 0
    # for k, m in enumerate(model.modules()):
    #     if isinstance(m, nn.BatchNorm2d):
    #         size = m.weight.data.numel()
    #         weight_copy = m.weight.data.abs().clone()
    #         _mask = weight_copy.gt(thre.cuda()).float().cuda()
    #         mask[index:(index + size)] = _mask.view(-1)
    #         # print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.format(k, _mask.shape[0], int(torch.sum(_mask))))
    #         index += size

    # print('Pre-processing Successful!')
    mask2 = bn.gt(thre).float().view(-1)
    return mask2


def clean_activate(model, percent):
    total = model.channel_size_all
    activate_time = torch.zeros(total)
    for data, target in validation_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        output = model(data)
        activate = model.getChannelActivate()
        no_activate_num = torch.sum(activate < 0)
        #print(no_activate_num)
        # activate_time += torch.where(activate > 0, 1, 0)
        activate_time += activate

    y, i = torch.sort(activate_time)
    thre_index = int(total * percent)
    thre = y[thre_index]

    mask = activate_time.gt(thre).float().view(-1)
    return mask


resume = args.save
print('==> resumeing from {} ... '.format(args.save))
checkpoint = torch.load(resume)
best_epoch = checkpoint['epoch']
print('EarlyBird Emerging Epoch: ', best_epoch)
model.load_state_dict(checkpoint['state_dict'])

percent_1 = 0.3 if 'EB-30' in args.save else 0.5 if 'EB-50' in args.save else 0.7 if 'EB-70' in args.save else setting_perc
best_mask = pruning(model, percent_1)

# print((best_mask == mask2).sum() == mask2.shape[0])


# -----------------
# size = best_mask.size(0)
# clean_activate_mask = clean_activate(model, percent_1)
# print('overlap rate of two masks: ', float(torch.sum(best_mask == clean_activate_mask)) / size)

# ------------------
size = best_mask.size(0)
print('Remanent Percent: {}%.'.format(int(torch.sum(best_mask == 1) * 100. / size)))

resume_bd = args.save_1
print('==> resumeing from {} ... '.format(args.save_1))
checkpoint_bd = torch.load(resume_bd)
best_epoch_bd = checkpoint_bd['epoch']
print('EarlyBird Emerging Epoch: ', best_epoch_bd)
model_bd.load_state_dict(checkpoint_bd['state_dict'])
percent_2 = 0.3 if 'EB-30' in args.save_1 else 0.5 if 'EB-50' in args.save_1 else 0.7 if 'EB-70' in args.save_1 else setting_perc

best_mask_bd = pruning(model_bd, percent_2)
print('Remanent Percent: {}%.'.format(int(torch.sum(best_mask_bd == 1) * 100. / size)))
print('overlap rate of two masks: ', float(torch.sum(best_mask == best_mask_bd)) / size)
# ----------------------

# ------------------
size = best_mask.size(0)
print('Remanent Percent: {}%.'.format(int(torch.sum(best_mask == 1) * 100. / size)))

resume_bd = args.save_2
print('==> resumeing from {} ... '.format(args.save_2))
checkpoint_bd = torch.load(resume_bd)
best_epoch_bd = checkpoint_bd['epoch']
print('EarlyBird Emerging Epoch: ', best_epoch_bd)
model_bd.load_state_dict(checkpoint_bd['state_dict'])
percent_3 = 0.3 if 'EB-30' in args.save_2 else 0.5 if 'EB-50' in args.save_2 else 0.7 if 'EB-70' in args.save_1 else setting_perc

best_mask_bd_1 = pruning(model_bd, percent_3)
print('Remanent Percent: {}%.'.format(int(torch.sum(best_mask_bd_1 == 1) * 100. / size)))
print('overlap rate of two masks: ', float(torch.sum(best_mask == best_mask_bd_1)) / size)
#print('overlap rate of two masks: ', float(torch.sum(best_mask_bd_1 == best_mask_bd)) / size)

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
