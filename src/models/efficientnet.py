import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch.utils import MemoryEfficientSwish

from models.layer import GeM, Mish


class ENet(nn.Module):
    def __init__(self, model_name='efficientnet-b3', n_channels=3, n_classes=None, pretrained=False):
        super(ENet, self).__init__()
        model_encoder = {
            # 'efficientnet-b0': {
            #     'model': EfficientNet.from_name('efficientnet-b0'),
            #     'pretrained_model': EfficientNet.from_pretrained('efficientnet-b0', advprop=True),
            #     'out_channels': 1280
            # },
            # 'efficientnet-b1': {
            #     'model': EfficientNet.from_name('efficientnet-b1'),
            #     'pretrained_model': EfficientNet.from_pretrained('efficientnet-b1', advprop=True),
            #     'out_channels': 1280
            # },
            # 'efficientnet-b2': {
            #     'model': EfficientNet.from_name('efficientnet-b2'),
            #     'pretrained_model': EfficientNet.from_pretrained('efficientnet-b2', advprop=True),
            #     'out_channels': 1408
            # },
            'efficientnet-b3': {
                'model': EfficientNet.from_name('efficientnet-b3'),
                'pretrained_model': EfficientNet.from_pretrained('efficientnet-b3', advprop=True),
                'out_channels': 1536
            },
            # 'efficientnet-b4': {
            #     'model': EfficientNet.from_name('efficientnet-b4'),
            #     'pretrained_model': EfficientNet.from_pretrained('efficientnet-b4', advprop=True),
            #     'out_channels': 1792
            # },
            # 'efficientnet-b5': {
            #     'model': EfficientNet.from_name('efficientnet-b5'),
            #     'pretrained_model': EfficientNet.from_pretrained('efficientnet-b5', advprop=True),
            #     'out_channels': 2048
            # },
            # 'efficientnet-b6': {
            #     'model': EfficientNet.from_name('efficientnet-b6'),
            #     'pretrained_model': EfficientNet.from_pretrained('efficientnet-b6', advprop=True),
            #     'out_channels': 2304
            # },
            # 'efficientnet-b7': {
            #     'model': EfficientNet.from_name('efficientnet-b7'),
            #     'pretrained_model': EfficientNet.from_pretrained('efficientnet-b7', advprop=True),
            #     'out_channels': 2560
            # },
        }

        if pretrained:
            self.model = model_encoder[model_name]['pretrained_model']
        else:
            self.model = model_encoder[model_name]['model']
        self.out_channels = model_encoder[model_name]['out_channels']

        self.model._conv_stem.in_channels = n_channels

        conv_stem = self.model._conv_stem
        if n_channels <= 3:
            self.model._conv_stem.weight.data[:, :n_channels, :, :] = conv_stem.weight.data[:, :n_channels, :, :]
        else:
            self.model._conv_stem.weight.data[:, :3, :, :] = conv_stem.weight.data
            self.model._conv_stem.weight.data[:, 3:n_channels, :, :] = conv_stem.weight.data[:, :int(n_channels - 3), :, :]

        self.model._avg_pooling = GeM()
        self.model._fc = nn.Sequential(
            nn.BatchNorm1d(self.out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        self.model._fc.out_channels = self.out_channels

        self._fc_gr = nn.Sequential(
                nn.BatchNorm1d(self.out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.Linear(in_features=self.out_channels, out_features=n_classes[0], bias=True)
        )
        self._fc_vo = nn.Sequential(
                nn.BatchNorm1d(self.out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.Linear(in_features=self.out_channels, out_features=n_classes[1], bias=True)
        )
        self._fc_co = nn.Sequential(
                nn.BatchNorm1d(self.out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.Linear(in_features=self.out_channels, out_features=n_classes[2], bias=True)
        )

    def forward(self, x):
        x = self.model(x)
        x_gr = self._fc_gr(x)
        x_vo = self._fc_vo(x)
        x_co = self._fc_co(x)
        return x_gr, x_vo, x_co
