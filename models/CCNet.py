import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch
from torch.nn import functional as F
from torch import Tensor
from typing import Callable, Optional

__all__ = ['CCNet']
model_urls = {
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
}

class Dropout(nn.Dropout):
    def __init__(self, p: float = 0.5, inplace: bool = False):
        """
        During training, randomly zeroes some of the elements of the input tensor with probability `p` using samples \
        from a Bernoulli distribution.

        :param p: probability of an element to be zeroed. Default: 0.5
        :param inplace: If set to ``True``, will do this operation in-place. Default: ``False``
        """
        super(Dropout, self).__init__(p=p, inplace=inplace)

    def profile_module(self, input: Tensor) -> (Tensor, float, float):
        input = self.forward(input)
        return input, 0.0, 0.0

class CA_layer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CA_layer, self).__init__()
        # global average pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel//reduction, kernel_size=(1, 1), bias=False),
            nn.Hardswish(),
            nn.Conv2d(channel//reduction, channel, kernel_size=(1, 1), bias=False),
            nn.Hardsigmoid()
        )

    def forward(self, x):
        y = self.fc(self.gap(x))
        return x*y.expand_as(x)

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class ParC_operator(nn.Module):
    def __init__(self, dim, type, global_kernel_size, use_pe=True, groups=1):
        super().__init__()
        self.type = type  # H or W
        self.dim = dim
        self.use_pe = use_pe
        self.global_kernel_size = global_kernel_size
        self.kernel_size = (global_kernel_size, 1) if self.type == 'H' else (1, global_kernel_size)
        self.gcc_conv = nn.Conv2d(dim, dim, kernel_size=self.kernel_size, groups=groups).weight
        if use_pe:
            if self.type=='H':
                torch.manual_seed(0)
                self.pe = nn.Parameter(torch.randn(1, dim, self.global_kernel_size, 1))
            elif self.type=='W':
                torch.manual_seed(0)
                self.pe = nn.Parameter(torch.randn(1, dim, 1, self.global_kernel_size))
            nn.init.trunc_normal_(self.pe, std=.02)

    def forward(self, x):
        b, c, h, w = x.size()
        if self.type == 'H':
            self.GCC = F.interpolate(self.gcc_conv, [h, 1], mode='bilinear', align_corners=True)
            self.PE = F.interpolate(self.pe, [h, 1], mode='bilinear', align_corners=True)
        elif self.type == 'W':
            self.GCC = F.interpolate(self.gcc_conv, [1, w], mode='bilinear', align_corners=True)
            self.PE = F.interpolate(self.pe, [1, w], mode='bilinear', align_corners=True)
        if self.use_pe:
            x = x + self.PE.expand(1, self.dim, h, w)

        x_cat = torch.cat((x, x[:, :, :-1, :]), dim=2) if self.type == 'H' else torch.cat((x, x[:, :, :, :-1]), dim=3)
        x = F.conv2d(x_cat, weight=self.GCC)

        return x


class ParC_block(nn.Module):

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        global_kernel_size = 96,
        use_pe = True,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(ParC_block, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.parc_H = ParC_operator(width//2, 'H', global_kernel_size, use_pe, groups = groups)
        self.parc_W = ParC_operator(width//2, 'W', global_kernel_size, use_pe, groups = groups)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        self.ffn = nn.Sequential(
            nn.Conv2d(512, 2, kernel_size=(1, 1), bias=True),
            nn.Hardswish(),
            Dropout(p=0.0),
            nn.Conv2d(2, 512, kernel_size=(1, 1), bias=True),
            Dropout(p=0.1)
        )
        self.ca = CA_layer(channel=512)

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv1(x)
        out = self.relu(out)
        out_H, out_W = torch.chunk(out, 2, dim=1)
        out_H, out_W = self.parc_H(out_H), self.parc_W(out_W)
        out_H, out_W = self.parc_W(out_H), self.parc_H(out_W)
        out = torch.cat((out_H, out_W), dim=1)
        out = self.relu(out)
        out = self.conv3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        out = out + self.ca(self.ffn(out))

        return out

class VGG(nn.Module):
    def __init__(self, features):
        super(VGG, self).__init__()
        self.features = features
        self.parc = ParC_block(512, 128)
        self.reg_layer = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.parc(x)
        x = F.upsample_bilinear(x, scale_factor=2)
        x = self.reg_layer(x)
        return torch.abs(x)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

cfg = {
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512]
}

def CCNet():
    """VGG 19-layer model (configuration "E")
        model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['E']))
    model.load_state_dict(model_zoo.load_url(model_urls['vgg19']), strict=False)
    return model

