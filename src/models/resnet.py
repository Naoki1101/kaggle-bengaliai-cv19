import torch
import torch.nn as nn
from torchvision import models

from models.layer import GeM, Mish


class ResNet(nn.Module):

    def __init__(self, model_name='resnet50', n_channels=3, n_classes=None, pretrained=False):
        super(ResNet, self).__init__()
        model_encoder = {
            'resnet18': {
                'model': models.resnet18,
                'out_channels': 512
            },
            'resnet34': {
                'model': models.resnet34,
                'out_channels': 512
            },
            'resnet50': {
                'model': models.resnet50,
                'out_channels': 2048
            },
            'resnet101': {
                'model': models.resnet101,
                'out_channels': 2048
            },
            'resnet152': {
                'model': models.resnet152,
                'out_channels': 2048
            },
            'resnext50_32x4d': {
                'model': models.resnext50_32x4d,
                'out_channels': 2048
            },
            'resnext101_32x8d': {
                'model': models.resnext101_32x8d,
                'out_channels': 2048
            },
            'wide_resnet50_2': {
                'model': models.wide_resnet50_2,
                'out_channels': 2048
            },
            'wide_resnet101_2': {
                'model': models.wide_resnet101_2,
                'out_channels': 2048
            },
        }
        self.model = model_encoder[model_name]['model'](pretrained=pretrained)
        self.out_channels = model_encoder[model_name]['out_channels']

        conv1 = self.model.conv1
        self.model.conv1 = nn.Conv2d(in_channels=n_channels,
                                     out_channels=conv1.out_channels,
                                     kernel_size=conv1.kernel_size,
                                     stride=conv1.stride,
                                     padding=conv1.padding,
                                     bias=conv1.bias)
        if n_channels <= 3:
            self.model.conv1.weight.data[:, :n_channels, :, :] = conv1.weight.data[:, :n_channels, :, :]
        else:
            self.model.conv1.weight.data[:, :3, :, :] = conv1.weight.data
            self.model.conv1.weight.data[:, 3:n_channels, :, :] = conv1.weight.data[:, :int(n_channels - 3), :, :]

        self.model.avgpool = GeM()
        self.model.fc = nn.Sequential(
            nn.BatchNorm1d(self.out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )

        self.fc_gr = nn.Sequential(
            nn.Linear(in_features=self.out_channels, out_features=n_classes[0], bias=True)
        )
        self.fc_gr2 = nn.Sequential(
            nn.Linear(in_features=sum(n_classes), out_features=n_classes[0], bias=True)
        )

        self.fc_vo = nn.Sequential(
            nn.Linear(in_features=self.out_channels, out_features=n_classes[1], bias=True)
        )
        self.fc_vo2 = nn.Sequential(
            nn.Linear(in_features=sum(n_classes), out_features=n_classes[1], bias=True)
        )

        self.fc_co = nn.Sequential(
            nn.Linear(in_features=self.out_channels, out_features=n_classes[2], bias=True)
        )
        self.fc_co2 = nn.Sequential(
            nn.Linear(in_features=sum(n_classes), out_features=n_classes[2], bias=True)
        )

    def forward(self, x):
        x = self.model(x)
        x_gr = self.fc_gr(x)
        x_vo = self.fc_vo(x)
        x_co = self.fc_co(x)

        # x = torch.cat((x_gr, x_vo, x_co), dim=1)
        # x_gr = self.fc_gr2(x)
        # x_vo = self.fc_vo2(x)
        # x_co = self.fc_co2(x)
        return x_gr, x_vo, x_co
