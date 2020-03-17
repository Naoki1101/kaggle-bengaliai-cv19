import torch
import torch.nn as nn
from torchvision import models

from models.layer import GeM, Mish


# https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes , stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_channels=3, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(num_channels, self.inplanes, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(self.inplanes, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = norm_layer(self.inplanes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(self.inplanes, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = norm_layer(self.inplanes)
        self.relu3 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=(stride, stride), stride=stride),
                conv1x1(self.inplanes, planes * block.expansion, stride=1),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


class ResNetD(nn.Module):
    def __init__(self, model_name, n_channels=3, n_classes=None):
        super(ResNetD, self).__init__()
        model_encoder = {
            'resnetd18': {
                'model': ResNet(block=BasicBlock,
                                layers=[2, 2, 2, 2],
                                num_channels=n_channels),
                'out_channels': 512
            },
            'resnetd34': {
                'model': ResNet(block=BasicBlock,
                                layers=[3, 4, 6, 3],
                                num_channels=n_channels),
                'out_channels': 512
            },
            'resnetd50': {
                'model': ResNet(block=Bottleneck,
                                layers=[3, 4, 6, 3],
                                num_channels=n_channels),
                'out_channels': 2048
            }
        }
        self.model = model_encoder[model_name]['model']
        self.out_channels = model_encoder[model_name]['out_channels']
        
        self.model.avg_pool = GeM()
        self.model.fc = nn.Sequential(
            nn.BatchNorm1d(self.out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        )

        self.fc_gr = nn.Sequential(
            # nn.Linear(in_features=self.out_channels, out_features=self.out_channels, bias=True),
            # nn.BatchNorm1d(self.out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            # nn.ReLU(),
            # nn.BatchNorm1d(self.out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            # nn.Dropout(p=0.5),
            nn.Linear(in_features=self.out_channels, out_features=n_classes[0], bias=True)
        )
        self.fc_vo = nn.Sequential(
            # nn.Linear(in_features=self.out_channels, out_features=self.out_channels, bias=True),
            # nn.BatchNorm1d(self.out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            # nn.ReLU(),
            # nn.BatchNorm1d(self.out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            # nn.Dropout(p=0.5),
            nn.Linear(in_features=self.out_channels, out_features=n_classes[1], bias=True)
        )
        self.fc_co = nn.Sequential(
            # nn.Linear(in_features=self.out_channels, out_features=self.out_channels, bias=True),
            # nn.BatchNorm1d(self.out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            # nn.ReLU(),
            # nn.BatchNorm1d(self.out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            # nn.Dropout(p=0.5),
            nn.Linear(in_features=self.out_channels, out_features=n_classes[2], bias=True)
        )

    def forward(self, x):
        x = self.model(x)
        x_gr = self.fc_gr(x)
        x_vo = self.fc_vo(x)
        x_co = self.fc_co(x)
        return x_gr, x_vo, x_co
