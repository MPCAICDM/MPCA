import torch
import torch.nn as nn


class _Encoder(nn.Module):
    def __init__(self, in_channels, hidden_layer_sizes, activation, pad_mode, bn, is_flatten=False, flatten_size=0):
        super(_Encoder, self).__init__()

        self.activation = activation
        self.hidden_layer_sizes = hidden_layer_sizes
        layers = []
        layers.append(nn.Conv2d(in_channels=in_channels, out_channels=hidden_layer_sizes[0], kernel_size=5, stride=2,
                      padding=2))
        layers.append(self.activation)
        if bn:
            layers.append(nn.BatchNorm2d(num_features=hidden_layer_sizes[0]))

        layers.append(nn.Conv2d(in_channels=hidden_layer_sizes[0], out_channels=hidden_layer_sizes[1], kernel_size=5, stride=2,
                      padding=2))
        layers.append(self.activation)
        if bn:
            layers.append(nn.BatchNorm2d(num_features=hidden_layer_sizes[1]))

        layers.append(nn.Conv2d(in_channels=hidden_layer_sizes[1], out_channels=hidden_layer_sizes[2], kernel_size=3, stride=2,
                      padding=pad_mode))
        layers.append(self.activation)
        if bn:
            layers.append(nn.BatchNorm2d(num_features=hidden_layer_sizes[2]))

        if is_flatten:
            h_channels = hidden_layer_sizes[2] * 4 * 4
            layers.append(nn.Flatten())
            layers.append(nn.Linear(h_channels, flatten_size))
            layers.append(self.activation)
        self.layers = nn.Sequential(*layers)
        print(self.layers)

    def forward(self, x):
        return self.layers(x)

class _Encoder_Linear(nn.Module):
    def __init__(self, in_channels, hidden_layer_sizes, activation, bn, last_activ=None,  is_flatten=False, flatten_size=None):
        super(_Encoder_Linear, self).__init__()

        self.activation = activation
        self.hidden_layer_sizes = hidden_layer_sizes
        layers = []
        if last_activ is None:
            for ic, oc in zip([in_channels] + hidden_layer_sizes[:-1], hidden_layer_sizes):
                layers.append(nn.Linear(in_features=ic, out_features=oc))
                layers.append(self.activation)
                if bn :
                    layers.append( nn.BatchNorm1d(num_features=oc))
        else:
            for ic, oc in zip([in_channels] + hidden_layer_sizes[:-1], hidden_layer_sizes):
                layers.append(nn.Linear(in_features=ic, out_features=oc))
                layers.append(self.activation)
            layers = layers[:-1]
            layers.append(last_activ)

        if is_flatten:
            h_channels = hidden_layer_sizes[-1]
            layers.append(nn.Linear(in_features=h_channels, out_features=flatten_size))
            layers.append(self.activation)

        self.layers = nn.Sequential(*layers)
        print(self.layers)

    def forward(self, x):
        return self.layers(x)

class _Decoder(nn.Module):
    def __init__(self, in_channels,out_channels, h_shape, hidden_layer_sizes, activation,pad_mode,bn, group=1,
                 last_channel=16, add_conv=False):
        super(_Decoder, self).__init__()

        self.activation = activation
        self.hidden_layer_sizes = hidden_layer_sizes
        self.h_shape = h_shape
        h_channels = h_shape[0]*h_shape[1]*h_shape[2]
        self.fc = nn.Linear(in_features=in_channels, out_features=h_channels)
        self.bnlayer = nn.BatchNorm2d(num_features=h_shape[0]) if bn else None
        layers = []
        layers.append(nn.ConvTranspose2d(in_channels=h_shape[0], out_channels=hidden_layer_sizes[1], kernel_size=3, stride=2,
                               padding=pad_mode, output_padding=pad_mode))
        layers.append(self.activation)
        if bn:
            layers.append(nn.BatchNorm2d(num_features=hidden_layer_sizes[1]))

        layers.append(
            nn.ConvTranspose2d(in_channels=hidden_layer_sizes[1], out_channels=hidden_layer_sizes[0], kernel_size=5,
                               stride=2,padding=2, output_padding=1))
        layers.append(self.activation)
        if bn:
            layers.append(nn.BatchNorm2d(num_features=hidden_layer_sizes[0]))

        self.group = group
        if add_conv:
            layers.append(nn.ConvTranspose2d(in_channels=hidden_layer_sizes[0], out_channels=last_channel, kernel_size=5, stride=2,
                          padding=2, output_padding=1))
            layers.append(self.activation)
            if bn:
                layers.append(nn.BatchNorm2d(num_features=last_channel))
            layers.append(nn.Conv2d(in_channels=last_channel, out_channels=out_channels*group, kernel_size=1,groups=group, bias=False))
            layers.append(self.activation)
        else:
            layers.append(nn.ConvTranspose2d(in_channels=hidden_layer_sizes[0], out_channels=out_channels * group, kernel_size=5,
                                   stride=2,
                                   padding=2, output_padding=1))
            layers.append(self.activation)
        self.layers = nn.Sequential(*layers)
        print(self.layers)

    def forward(self, x):
        h = self.fc(x)
        h = self.activation(h)
        h = h.view(-1, *self.h_shape)
        if self.bnlayer is not None:
            h = self.bnlayer(h)
        return self.layers(h)

class _Decoder_Linear(nn.Module):
    def __init__(self, in_channels,out_channels, hidden_layer_sizes, activation, bn):
        super(_Decoder_Linear, self).__init__()

        self.activation = activation
        self.hidden_layer_sizes = hidden_layer_sizes
        hidden_layer_sizes = hidden_layer_sizes[::-1]
        layers = []
        for ic, oc in zip([in_channels] + hidden_layer_sizes[:-1], hidden_layer_sizes):
            layers.append(nn.Linear(in_features=ic, out_features=oc))
            layers.append(self.activation)
            if bn:
                layers.append(nn.BatchNorm1d(num_features=oc))
        layers.append(nn.Linear(in_features=hidden_layer_sizes[-1], out_features=out_channels))
        layers.append(self.activation)
        self.deconv = nn.Sequential(*layers)
        print(self.deconv)

    def forward(self, x):
        h = self.deconv(x)
        return h

class CAE_backbone(nn.Module):
    def __init__(self, input_shape, hidden_layer_sizes, bn, flatten_size=128):
        super(CAE_backbone, self).__init__()
        c, h, w = input_shape
        pad_mode = 1 if h % 8 == 0 else 0

        self.encoder = _Encoder(in_channels=c, hidden_layer_sizes=hidden_layer_sizes,
                                activation=nn.Tanh(), pad_mode=pad_mode, bn=bn, is_flatten=True,
                                flatten_size=flatten_size)

        h_shape = (hidden_layer_sizes[2], h // 8, w // 8)
        h_channels = flatten_size

        self.decoder = _Decoder(in_channels=h_channels, out_channels=c, h_shape=h_shape,
                                hidden_layer_sizes=hidden_layer_sizes, activation=nn.Tanh(),
                                pad_mode=pad_mode, add_conv=False, bn=bn)

    def forward(self, x):
        y = self.encoder(x)
        y = y.view(y.shape[0], -1)
        x_r = self.decoder(y)
        return x_r


class AE_backbone(nn.Module):
    def __init__(self, input_shape, hidden_layer_sizes, bn, flatten_size=128):
        super(AE_backbone, self).__init__()
        c = input_shape
        self.encoder = _Encoder_Linear(in_channels=c, hidden_layer_sizes=hidden_layer_sizes,
                                       activation=nn.ReLU(), bn=bn, is_flatten=True, flatten_size=flatten_size)

        h_channels = flatten_size

        self.decoder = _Decoder_Linear(in_channels=h_channels, out_channels=c,
                                       hidden_layer_sizes=hidden_layer_sizes, activation=nn.ReLU(),
                                       bn=bn)

    def forward(self, x):
        y = self.encoder(x)
        x_r = self.decoder(y)
        return x_r
