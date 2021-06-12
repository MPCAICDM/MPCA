from functools import reduce
from operator import mul
from typing import Tuple

import torch
import torch.nn as nn
from loss_functions.rsr_loss import RSRLoss


class _Encoder(nn.Module):
    def __init__(self, in_channels, hidden_layer_sizes, activation, pad_mode):
        super(_Encoder, self).__init__()

        self.activation = activation
        self.hidden_layer_sizes = hidden_layer_sizes
        self.is_bn = True
        if self.is_bn:
            self.layers = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=hidden_layer_sizes[0], kernel_size=5, stride=2,
                          padding=2), # 14*14
                self.activation,
                nn.BatchNorm2d(num_features=hidden_layer_sizes[0]),

                nn.Conv2d(in_channels=hidden_layer_sizes[0], out_channels=hidden_layer_sizes[1], kernel_size=5, stride=2,
                          padding=2), # 7 * 7
                self.activation,
                nn.BatchNorm2d(num_features=hidden_layer_sizes[1]),

                nn.Conv2d(in_channels=hidden_layer_sizes[1], out_channels=hidden_layer_sizes[2], kernel_size=3, stride=2,
                          padding=pad_mode), # 3 * 3
                self.activation,
                nn.BatchNorm2d(num_features=hidden_layer_sizes[2])
            )
        else:
            self.layers = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=hidden_layer_sizes[0], kernel_size=5, stride=2,
                          padding=2),  # 14*14
                self.activation,
                #nn.BatchNorm2d(num_features=hidden_layer_sizes[0]),

                nn.Conv2d(in_channels=hidden_layer_sizes[0], out_channels=hidden_layer_sizes[1], kernel_size=5,
                          stride=2,
                          padding=2),  # 7 * 7
                self.activation,
                #nn.BatchNorm2d(num_features=hidden_layer_sizes[1]),

                nn.Conv2d(in_channels=hidden_layer_sizes[1], out_channels=hidden_layer_sizes[2], kernel_size=3,
                          stride=2,
                          padding=pad_mode),  # 3 * 3
                self.activation,
                #nn.BatchNorm2d(num_features=hidden_layer_sizes[2])
            )

    def forward(self, x):
        return self.layers(x)

class _Decoder(nn.Module):
    def __init__(self, in_channels,out_channels, h_shape, hidden_layer_sizes, activation, pad_mode):
        super(_Decoder, self).__init__()

        self.is_bn = True

        self.activation = activation
        self.hidden_layer_sizes = hidden_layer_sizes
        self.h_shape = h_shape
        h_channels = h_shape[0]*h_shape[1]*h_shape[2]
        self.fc = nn.Linear(in_features=in_channels, out_features=h_channels)
        self.bn = nn.BatchNorm2d(num_features=h_shape[0])
        if self.is_bn:
            self.deconv1 = nn.Sequential(
                nn.ConvTranspose2d(in_channels=h_shape[0], out_channels=hidden_layer_sizes[1], kernel_size=3, stride=2,
                                   padding=pad_mode, output_padding=pad_mode),
                self.activation,
                nn.BatchNorm2d(num_features=hidden_layer_sizes[1])
            )
            self.deconv2 = nn.Sequential(
                nn.ConvTranspose2d(in_channels=hidden_layer_sizes[1], out_channels=hidden_layer_sizes[0], kernel_size=5,
                                   stride=2,padding=2, output_padding=1),
                self.activation,
                nn.BatchNorm2d(num_features=hidden_layer_sizes[0])
            )
            self.deconv3 = nn.Sequential(
                nn.ConvTranspose2d(in_channels=hidden_layer_sizes[0], out_channels=out_channels, kernel_size=5, stride=2,
                          padding=2, output_padding=1),
                self.activation,
               # nn.Conv2d(in_channels=16, out_channels=out_channels, kernel_size=1, bias=False),
               # self.activation,
            )
        else:
            self.deconv1 = nn.Sequential(
                nn.ConvTranspose2d(in_channels=h_shape[0], out_channels=hidden_layer_sizes[1], kernel_size=3, stride=2,
                                   padding=pad_mode, output_padding=pad_mode),
                self.activation,
                # nn.BatchNorm2d(num_features=hidden_layer_sizes[1])
            )
            self.deconv2 = nn.Sequential(
                nn.ConvTranspose2d(in_channels=hidden_layer_sizes[1], out_channels=hidden_layer_sizes[0], kernel_size=5,
                                   stride=2, padding=2, output_padding=1),
                self.activation,
                # nn.BatchNorm2d(num_features=hidden_layer_sizes[0])
            )
            self.deconv3 = nn.Sequential(
                nn.ConvTranspose2d(in_channels=hidden_layer_sizes[0], out_channels=out_channels, kernel_size=5,
                                   stride=2,
                                   padding=2, output_padding=1),
                self.activation,
                # nn.Conv2d(in_channels=16, out_channels=out_channels, kernel_size=1, bias=False),
                # self.activation,
            )

    def forward(self, x):
        h = self.fc(x)
        h = self.activation(h)
        h = h.view(-1, *self.h_shape)
        h = self.bn(h)
        #print(h.shape)
        h = self.deconv1(h)
        h = self.deconv2(h)
        h = self.deconv3(h)
        return h

def l2_norm(input, dim):
    norm = torch.norm(input, 2, dim, True)
    output = torch.div(input, norm)
    return output

class _Decoder_Linear(nn.Module):
    def __init__(self, in_channels,out_channels, hidden_layer_sizes, activation, bn):
        super(_Decoder_Linear, self).__init__()

        self.activation = activation
        self.hidden_layer_sizes = hidden_layer_sizes

        if bn:
            self.deconv = nn.Sequential(
                nn.Linear(in_features=in_channels, out_features=hidden_layer_sizes[2]),  # 14*14
                self.activation,
                nn.BatchNorm1d(num_features=hidden_layer_sizes[2]),

                nn.Linear(in_features=hidden_layer_sizes[2], out_features=hidden_layer_sizes[1]),  # 14*14
                self.activation,
                nn.BatchNorm1d(num_features=hidden_layer_sizes[1]),

                nn.Linear(in_features=hidden_layer_sizes[1], out_features=hidden_layer_sizes[0]),  # 14*14
                self.activation,
                nn.BatchNorm1d(num_features=hidden_layer_sizes[0]),

                nn.Linear(in_features=hidden_layer_sizes[0], out_features=out_channels),  # 14*14
                self.activation,
            )
        else:
            self.deconv = nn.Sequential(
                nn.Linear(in_features=in_channels, out_features=hidden_layer_sizes[2]),  # 14*14
                self.activation,

                nn.Linear(in_features=hidden_layer_sizes[2], out_features=hidden_layer_sizes[1]),  # 14*14
                self.activation,

                nn.Linear(in_features=hidden_layer_sizes[1], out_features=hidden_layer_sizes[0]),  # 14*14
                self.activation,

                nn.Linear(in_features=hidden_layer_sizes[0], out_features=out_channels),  # 14*14
                self.activation,
            )
            print('nobn')

    def forward(self, x):
        h = self.deconv(x)
        return h

class _Encoder_Linear(nn.Module):
    def __init__(self, in_channels, hidden_layer_sizes, activation, bn):
        super(_Encoder_Linear, self).__init__()

        self.activation = activation
        self.hidden_layer_sizes = hidden_layer_sizes
        if bn:
            self.layers = nn.Sequential(
                nn.Linear(in_features=in_channels, out_features=hidden_layer_sizes[0]),  # 14*14
                self.activation,
                nn.BatchNorm1d(num_features=hidden_layer_sizes[0]),

                nn.Linear(in_features=hidden_layer_sizes[0], out_features=hidden_layer_sizes[1]),  # 14*14
                self.activation,
                nn.BatchNorm1d(num_features=hidden_layer_sizes[1]),

                nn.Linear(in_features=hidden_layer_sizes[1], out_features=hidden_layer_sizes[2]),  # 14*14
                self.activation,
                nn.BatchNorm1d(num_features=hidden_layer_sizes[2]),
            )
        else:
            self.layers = nn.Sequential(
                nn.Linear(in_features=in_channels, out_features=hidden_layer_sizes[0]),  # 14*14
                self.activation,

                nn.Linear(in_features=hidden_layer_sizes[0], out_features=hidden_layer_sizes[1]),  # 14*14
                self.activation,

                nn.Linear(in_features=hidden_layer_sizes[1], out_features=hidden_layer_sizes[2]),  # 14*14
                self.activation,
            )
            print('nobn')


    def forward(self, x):
        return self.layers(x)


class RSRAE(nn.Module):
    def __init__(self, input_shape, hidden_layer_sizes, z_channels):
        super(RSRAE, self).__init__()
        h, w, c = input_shape
        pad_mode = 1 if h % 8 == 0 else 0
        self.encoder = _Encoder(in_channels=c, hidden_layer_sizes=hidden_layer_sizes, activation=nn.Tanh(), pad_mode=pad_mode)

        h_shape = (hidden_layer_sizes[2], h//8, w//8)
        h_channels = h_shape[0]*h_shape[1]*h_shape[2]

        self.A = torch.nn.Parameter(torch.randn(h_channels, z_channels))
        self.renorm = False

        self.decoder = _Decoder(in_channels=z_channels, out_channels=c, h_shape=h_shape,pad_mode=pad_mode,
                                hidden_layer_sizes=hidden_layer_sizes,activation=nn.Tanh())

    def forward(self, x):
        y = self.encoder(x)
        y = y.view(y.shape[0], -1)
        y_rsr = torch.matmul(y, self.A)
        if self.renorm:
            z = l2_norm(y_rsr, -1)
        else:
            z = y_rsr
        x_r = self.decoder(z)
        #print(y.shape, y_rsr.shape, z.shape, x_r.shape)
        return y, y_rsr, z, x_r

class RSRAE_Linear(nn.Module):
    def __init__(self, input_shape, hidden_layer_sizes, z_channels, bn):
        super(RSRAE_Linear, self).__init__()
        c = input_shape
        self.encoder = _Encoder_Linear(in_channels=c, hidden_layer_sizes=hidden_layer_sizes,
                                       activation=nn.ReLU(),bn=bn)
        self.bn = bn
        self.A = torch.nn.Parameter(torch.randn(hidden_layer_sizes[2], z_channels))
        self.renorm = False

        self.decoder = _Decoder_Linear(in_channels=z_channels, out_channels=c,bn=bn,
                                hidden_layer_sizes=hidden_layer_sizes,activation=nn.ReLU())

    def forward(self, x):
        y = self.encoder(x)
        y = y.view(y.shape[0], -1)
        y_rsr = torch.matmul(y, self.A)
        if self.renorm:
            z = l2_norm(y_rsr, -1)
        else:
            z = y_rsr
        x_r = self.decoder(z)
        #print(y.shape, y_rsr.shape, z.shape, x_r.shape)
        return y, y_rsr, z, x_r




