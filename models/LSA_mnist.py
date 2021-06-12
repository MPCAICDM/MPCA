from functools import reduce
from operator import mul

import torch
import torch.nn as nn

from models.blocks.LSA_blocks import DownsampleBlock, UpsampleBlock
from models.blocks.estimator_1D import Estimator1D

class Encoder(nn.Module):
    def __init__(self, input_shape, code_length):
        super(Encoder, self).__init__()

        self.input_shape = input_shape
        self.code_length = code_length

        c, h, w = input_shape

        activation_fn = nn.LeakyReLU()

        self.conv = nn.Sequential(
            DownsampleBlock(channel_in=c, channel_out=32, activation_fn=activation_fn),
            DownsampleBlock(channel_in=32, channel_out=64, activation_fn=activation_fn)
        )
        self.conv_feature_shape = (64, h//4, w//4)

        self.fc = nn.Sequential(
            nn.Linear(in_features=reduce(mul, self.conv_feature_shape), out_features=64),
            nn.BatchNorm1d(num_features=64),
            activation_fn,
            nn.Linear(in_features=64, out_features=code_length),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(len(x), -1)
        x = self.fc(x)
        return x


class Decoder(nn.Module):
    def __init__(self, code_length, conv_feature_shape, out_shape):
        super(Decoder, self).__init__()

        self.code_length = code_length
        self.out_shape = out_shape
        self.conv_feature_shape = conv_feature_shape

        activation_fn = nn.LeakyReLU()

        self.fc = nn.Sequential(
            nn.Linear(in_features=code_length, out_features=64),
            nn.BatchNorm1d(num_features=64),
            activation_fn,
            nn.Linear(in_features=64, out_features=reduce(mul, self.conv_feature_shape)),
            nn.BatchNorm1d(num_features=reduce(mul, self.conv_feature_shape)),
            activation_fn
        )

        self.deconv = nn.Sequential(
            UpsampleBlock(channel_in=64, channel_out=32, activation_fn=activation_fn),
            UpsampleBlock(channel_in=32, channel_out=16, activation_fn=activation_fn),
            nn.Conv2d(in_channels=16, out_channels=out_shape[0], kernel_size=1, bias=False),
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(len(x), *self.conv_feature_shape)
        x = self.deconv(x)
        return x


class DecoderGroup(nn.Module):
    def __init__(self, code_length, conv_feature_shape, out_shape, group=2):
        super(DecoderGroup, self).__init__()

        self.code_length = code_length
        self.out_shape = out_shape
        self.conv_feature_shape = conv_feature_shape

        activation_fn = nn.LeakyReLU()

        self.fc = nn.Sequential(
            nn.Linear(in_features=code_length, out_features=64),
            nn.BatchNorm1d(num_features=64),
            activation_fn,
            nn.Linear(in_features=64, out_features=reduce(mul, self.conv_feature_shape)),
            nn.BatchNorm1d(num_features=reduce(mul, self.conv_feature_shape)),
            activation_fn
        )

        self.deconv = nn.Sequential(
            UpsampleBlock(channel_in=64, channel_out=32, activation_fn=activation_fn),
            UpsampleBlock(channel_in=32, channel_out=16, activation_fn=activation_fn),
            nn.Conv2d(in_channels=16, out_channels=group, kernel_size=1,groups=group, bias=False),
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(len(x), *self.conv_feature_shape)
        x = self.deconv(x)
        return x


class LSA_MNIST(nn.Module):
    def __init__(self, input_shape, code_length, cpd_channels):
        super(LSA_MNIST, self).__init__()

        self.input_shape = input_shape
        self.code_length = code_length
        self.cpd_channels = cpd_channels

        self.encoder = Encoder(
            input_shape=input_shape,
            code_length = code_length
        )

        self.decoder = Decoder(
            code_length=code_length,
            conv_feature_shape=self.encoder.conv_feature_shape,
            out_shape=input_shape
        )

        self.estimator = Estimator1D(
            code_length=code_length,
            fm_list=[32, 32, 32, 32],
            cpd_channels=cpd_channels
        )


    def forward(self, x):
        z = self.encoder(x)
        z_dist = self.estimator(z)
        x_r = self.decoder(z)
        x_r= x_r.view(-1, *self.input_shape)
        return x_r, z, z_dist


class LSA_MNIST_DOUBLE(nn.Module):
    def __init__(self, input_shape, code_length, cpd_channels, group):
        super(LSA_MNIST_DOUBLE, self).__init__()

        self.input_shape = input_shape
        self.code_length = code_length
        self.cpd_channels = cpd_channels

        self.encoder = Encoder(
            input_shape=input_shape,
            code_length = code_length
        )

        self.decoder1 = DecoderGroup(
            code_length=code_length,
            conv_feature_shape=self.encoder.conv_feature_shape,
            out_shape=input_shape,
            group=group
        )
        self.group = group
        self.tanh = nn.Tanh() # need sigmoid


    def forward(self, x):
        z = self.encoder(x)
        x_r1 = self.decoder1(z)
        x_r1 = x_r1.view(-1, self.group, *self.input_shape[1:])
        #print(x_r1.shape)
        x_r = torch.chunk(x_r1, self.group, dim=1)
        return x_r,  x_r1.mean(dim=1, keepdim=True)