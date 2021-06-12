import torch
import torch.nn as nn
from models.blocks.transform_sos import TinvSOS
from models.LSA_mnist import Encoder, Decoder


class _Encoder(nn.Module):
    def __init__(self, in_channels, hidden_layer_sizes, activation):
        super(_Encoder, self).__init__()

        self.activation = activation
        self.hidden_layer_sizes = hidden_layer_sizes
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
                      padding=0), # 3 * 3
            self.activation,
            nn.BatchNorm2d(num_features=hidden_layer_sizes[2])
        )

    def forward(self, x):
        return self.layers(x)

class _Decoder(nn.Module):
    def __init__(self, in_channels,out_channels, h_shape, hidden_layer_sizes, activation, group=1, last_channel=16):
        super(_Decoder, self).__init__()

        self.activation = activation
        self.hidden_layer_sizes = hidden_layer_sizes
        self.h_shape = h_shape
        h_channels = h_shape[0]*h_shape[1]*h_shape[2]
        self.fc = nn.Linear(in_features=in_channels, out_features=h_channels)
        self.bn = nn.BatchNorm2d(num_features=h_shape[0])
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=h_shape[0], out_channels=hidden_layer_sizes[1], kernel_size=3, stride=2,
                               padding=0, output_padding=0),
            self.activation,
            nn.BatchNorm2d(num_features=hidden_layer_sizes[1])
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=hidden_layer_sizes[1], out_channels=hidden_layer_sizes[0], kernel_size=5,
                               stride=2,padding=2, output_padding=1),
            self.activation,
            nn.BatchNorm2d(num_features=hidden_layer_sizes[0])
        )
        self.group = group
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=hidden_layer_sizes[0], out_channels=last_channel, kernel_size=5, stride=2,
                      padding=2, output_padding=1),
            self.activation,
            #nn.BatchNorm2d(num_features=last_channel),
            #nn.Conv2d(in_channels=last_channel, out_channels=out_channels*group, kernel_size=1,groups=group, bias=False),
            #self.activation,
        )

    def forward(self, x):
        h = self.fc(x)
        h = self.activation(h)
        h = h.view(-1, *self.h_shape)
        h = self.bn(h)
        h = self.deconv1(h)
        h = self.deconv2(h)
        h = self.deconv3(h)
        return h

class MTQ_MNIST(nn.Module):
    def __init__(self, num_blocks, es_channels, input_shape, hidden_layer_sizes, backbone, code_length=64):
        super(MTQ_MNIST, self).__init__()
        c, h, w = input_shape
        if backbone == 'LSA':
            self.encoder = Encoder(
                input_shape=input_shape,
                code_length=code_length
            )
            self.h_channels = code_length
            self.decoder = Decoder(
                code_length=code_length,
                conv_feature_shape=self.encoder.conv_feature_shape,
                out_shape=input_shape
            )
            self.estimator = TinvSOS(num_blocks, code_length, es_channels)
        else:
            h_shape = (hidden_layer_sizes[2], h // 8, w // 8)
            self.h_channels = h_channels = h_shape[0] * h_shape[1] * h_shape[2]

            self.encoder = _Encoder(in_channels=c, hidden_layer_sizes=hidden_layer_sizes, activation=nn.Tanh())

            #self.A = torch.nn.Parameter(torch.randn(h_channels, z_channels))
            #self.fc = nn.Linear()
            self.estimator = TinvSOS(num_blocks, h_channels, es_channels)

            self.decoder = _Decoder(in_channels=h_channels, out_channels=c, h_shape=h_shape,
                                    last_channel=1,
                                    hidden_layer_sizes=hidden_layer_sizes,activation=nn.Tanh())


    def forward(self, x):
        y = self.encoder(x)
        z = y.view(y.shape[0], -1)
        s, log_jacob_T_inverse = self.estimator(z)
        x_r = self.decoder(z)
        return  x_r, z, s, log_jacob_T_inverse
