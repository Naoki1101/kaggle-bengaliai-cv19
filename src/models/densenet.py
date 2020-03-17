import torch.nn as nn
from torchvision import models


class DenseNet(nn.Module):

    def __init__(self, model_name='densenet121', n_channels=3, n_classes=None, pretrained=False):
        super(DenseNet, self).__init__()
        model_encoder = {
            'densenet121': {
                'model': models.densenet121,
                'out_channels': 1024
            }
        }
        self.model = model_encoder[model_name]['model'](pretrained=pretrained)
        self.out_channels = model_encoder[model_name]['out_channels']

        conv0 = self.model.features.conv0
        self.model.features.conv0 = nn.Conv2d(in_channels=n_channels,
                                              out_channels=conv0.out_channels,
                                              kernel_size=conv0.kernel_size,
                                              stride=conv0.stride,
                                              padding=conv0.padding,
                                              bias=conv0.bias)
        if n_channels <= 3:
            self.model.features.conv0.weight.data[:, :n_channels, :, :] = conv0.weight.data[:, :n_channels, :, :]
        else:
            self.model.features.conv0.weight.data[:, :3, :, :] = conv0.weight.data
            self.model.features.conv0.weight.data[:, 3:n_channels, :, :] = conv0.weight.data[:, :int(n_channels - 3), :, :]


        self.model.classifier = nn.Sequential(
            nn.BatchNorm1d(self.out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            # nn.Dropout(p=0.25)
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