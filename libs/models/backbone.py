import torch
import torch.nn as nn
import math

from collections import OrderedDict

def conv_bn(inp, oup, kernel=1, bias=False, dilation=1):

    pad = kernel // 2
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size=kernel, padding=pad, bias=bias, dilation=dilation),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu= nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.relu(self.bn2(out))

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out

class ResBlock(nn.Module):

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.relu= nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.relu(self.bn2(out))

        out += residual
        out = self.relu(out)

        return out


class Encoder(nn.Module):
    def __init__(self, block, layers, inplanes=[64, 128, 256, 512], inp=3):
        self.inplanes = 64
        super(Encoder, self).__init__()

        assert len(inplanes) == 4

        self.conv1 = nn.Conv2d(inp, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, inplanes[0], layers[0])
        self.layer2 = self._make_layer(block, inplanes[1], layers[1], stride=2)
        self.layer3 = self._make_layer(block, inplanes[2], layers[2], stride=2)
        self.layer4 = self._make_layer(block, inplanes[3], layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def load_param(self, weight):

        assert isinstance(weight, OrderedDict)
        state = self.state_dict()
        for key in weight:
            if key in state and state[key].shape == weight[key].shape:
                state[key][...] = weight[key][...]
            else:
                print('ignore mismatched block %s during initialization' % key)

        self.load_state_dict(state)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return x1, x2, x3, x4

class Refine(nn.Module):

    def __init__(self, up_dim, down_dim, out_dim):
        super(Refine, self).__init__()

        self.upstream = nn.Sequential(
            nn.Conv2d(up_dim, out_dim, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_dim),
            ResBlock(out_dim, out_dim)
        )

        self.downstream = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(down_dim, out_dim, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_dim)
        )

        self.relu = nn.ReLU(inplace=True)

        self.conv = nn.Sequential(
            ResBlock(out_dim, out_dim),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
        )

    def forward(self, up, down):

        out = self.relu(self.upstream(up) + self.downstream(down))

        return self.conv(out)

class Decoder(nn.Module):

    def __init__(self, in_dim, inplane, up_dims):
        super(Decoder, self).__init__()

        assert len(up_dims) == 3

        self.conv = conv_bn(in_dim, inplane)

        self.refine2 = Refine(up_dims[0], inplane, inplane)
        self.refine3 = Refine(up_dims[1], inplane, inplane)
        self.refine4 = Refine(up_dims[2], inplane, inplane)

    def forward(self, x1, x2, x3, x4):

        out = self.conv(x4)
        out3 = self.refine4(x3, out)
        out2 = self.refine3(x2, out3)
        out1 = self.refine2(x1, out2)

        return out1, out2, out3