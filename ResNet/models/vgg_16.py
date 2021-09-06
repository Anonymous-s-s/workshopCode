import math

import torch
import torch.nn as nn

__all__ = ['vgg_16']

defaultcfg = {
    11: [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    13: [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    16: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    19: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
}


class vgg_16(nn.Module):
    def __init__(self, dataset='cifar10', depth=16, init_weights=True, cfg=None):
        super(vgg_16, self).__init__()
        self.layer_conv2d_1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(64),
                                            nn.ReLU(inplace=True))

        self.layer_conv2d_2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
                                            nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        self.layer_conv2d_3 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
                                            nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.layer_conv2d_4 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
                                            nn.BatchNorm2d(128), nn.ReLU(inplace=True))

        self.layer_conv2d_5 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
                                            nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.layer_conv2d_6 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
                                            nn.BatchNorm2d(256), nn.ReLU(inplace=True))

        self.layer_conv2d_7 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
                                            nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.layer_conv2d_8 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
                                            nn.BatchNorm2d(512), nn.ReLU(inplace=True))

        self.layer_conv2d_9 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
                                            nn.BatchNorm2d(512), nn.ReLU(inplace=True))

        self.layer_conv2d_10 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
                                             nn.BatchNorm2d(512), nn.ReLU(inplace=True))

        self.layer_conv2d_11 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
                                             nn.BatchNorm2d(512), nn.ReLU(inplace=True))

        self.layer_conv2d_12 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
                                             nn.BatchNorm2d(512), nn.ReLU(inplace=True))

        self.layer_conv2d_13 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
                                             nn.BatchNorm2d(512), nn.ReLU(inplace=True))

        self.channel_size_all = 64 * 2 + 128 * 2 + 256 * 3 + 512 * 6  # conv num ->13
        self.channel_act_all = torch.zeros(self.channel_size_all)
        if dataset == 'cifar10':
            num_classes = 10
        elif dataset == 'cifar100':
            num_classes = 100

        self.classifier = nn.Linear(512, num_classes)
        if init_weights:
            self._initialize_weights()

    def forward(self, x):

        index = 0

        # x.shape=[batchsize,channel,H,W]
        x = self.layer_conv2d_1(x)
        channel_activate = torch.gt(x, 0).sum(axis=(0, 2, 3)) / (x.shape[0] * x.shape[2] * x.shape[3])
        self.channel_act_all[index:(index + x.shape[1])] = channel_activate
        index += x.shape[1]
        x = self.layer_conv2d_2(x)
        channel_activate = torch.gt(x, 0).sum(axis=(0, 2, 3)) / (x.shape[0] * x.shape[2] * x.shape[3])
        self.channel_act_all[index:(index + x.shape[1])] = channel_activate
        index += x.shape[1]

        x = nn.MaxPool2d(kernel_size=2, stride=2)(x)

        x = self.layer_conv2d_3(x)
        channel_activate = torch.gt(x, 0).sum(axis=(0, 2, 3)) / (x.shape[0] * x.shape[2] * x.shape[3])
        self.channel_act_all[index:(index + x.shape[1])] = channel_activate
        index += x.shape[1]
        x = self.layer_conv2d_4(x)
        channel_activate = torch.gt(x, 0).sum(axis=(0, 2, 3)) / (x.shape[0] * x.shape[2] * x.shape[3])
        self.channel_act_all[index:(index + x.shape[1])] = channel_activate
        index += x.shape[1]

        x = nn.MaxPool2d(kernel_size=2, stride=2)(x)

        x = self.layer_conv2d_5(x)
        channel_activate = torch.gt(x, 0).sum(axis=(0, 2, 3)) / (x.shape[0] * x.shape[2] * x.shape[3])
        self.channel_act_all[index:(index + x.shape[1])] = channel_activate
        index += x.shape[1]
        x = self.layer_conv2d_6(x)
        channel_activate = torch.gt(x, 0).sum(axis=(0, 2, 3)) / (x.shape[0] * x.shape[2] * x.shape[3])
        self.channel_act_all[index:(index + x.shape[1])] = channel_activate
        index += x.shape[1]
        x = self.layer_conv2d_7(x)
        channel_activate = torch.gt(x, 0).sum(axis=(0, 2, 3)) / (x.shape[0] * x.shape[2] * x.shape[3])
        self.channel_act_all[index:(index + x.shape[1])] = channel_activate
        index += x.shape[1]

        x = nn.MaxPool2d(kernel_size=2, stride=2)(x)

        x = self.layer_conv2d_8(x)
        channel_activate = torch.gt(x, 0).sum(axis=(0, 2, 3)) / (x.shape[0] * x.shape[2] * x.shape[3])
        self.channel_act_all[index:(index + x.shape[1])] = channel_activate
        index += x.shape[1]
        x = self.layer_conv2d_9(x)
        channel_activate = torch.gt(x, 0).sum(axis=(0, 2, 3)) / (x.shape[0] * x.shape[2] * x.shape[3])
        self.channel_act_all[index:(index + x.shape[1])] = channel_activate
        index += x.shape[1]
        x = self.layer_conv2d_10(x)
        channel_activate = torch.gt(x, 0).sum(axis=(0, 2, 3)) / (x.shape[0] * x.shape[2] * x.shape[3])
        self.channel_act_all[index:(index + x.shape[1])] = channel_activate
        index += x.shape[1]

        x = nn.MaxPool2d(kernel_size=2, stride=2)(x)

        x = self.layer_conv2d_11(x)
        channel_activate = torch.gt(x, 0).sum(axis=(0, 2, 3)) / (x.shape[0] * x.shape[2] * x.shape[3])
        self.channel_act_all[index:(index + x.shape[1])] = channel_activate
        index += x.shape[1]
        x = self.layer_conv2d_12(x)
        channel_activate = torch.gt(x, 0).sum(axis=(0, 2, 3)) / (x.shape[0] * x.shape[2] * x.shape[3])
        self.channel_act_all[index:(index + x.shape[1])] = channel_activate
        index += x.shape[1]
        x = self.layer_conv2d_13(x)
        channel_activate = torch.gt(x, 0).sum(axis=(0, 2, 3)) / (x.shape[0] * x.shape[2] * x.shape[3])
        self.channel_act_all[index:(index + x.shape[1])] = channel_activate
        index += x.shape[1]

        x = nn.AvgPool2d(2)(x)
        x = x.view(x.size(0), -1)
        y = self.classifier(x)
        return y

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def getChannelActivate(self):
        return self.channel_act_all


if __name__ == '__main__':
    net = vgg_16(dataset='cifar10')
    # x = torch.FloatTensor(16, 3, 40, 40)
    x = torch.randn([16, 3, 40, 40])
    y = net(x)
    print(y.data.shape)
    print(net.getChannelActivate().shape)
