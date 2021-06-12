import torch
import torch.nn as nn
from models.blocks.estimator_1D import Estimator1D
from models.blocks.LSA_blocks import DownsampleBlock, UpsampleBlock, ResidualBlock
from functools import reduce
from operator import mul

class _Encoder_LSA(nn.Module):
    def __init__(self, in_channels, hidden_layer_sizes, activation, pad_mode, bn, is_flatten=False, flatten_size=0):
        super(_Encoder_LSA, self).__init__()
        self.conv = nn.Sequential(
            #nn.Conv2d(in_channels=in_channels, out_channels=hidden_layer_sizes[0], kernel_size=3, bias=False),
            #activation,
            #ResidualBlock(channel_in=32, channel_out=32, activation_fn=activation, use_bn=False),
            DownsampleBlock(channel_in=in_channels, channel_out=32, activation_fn=activation,  use_bn=False),
            DownsampleBlock(channel_in=32, channel_out=64, activation_fn=activation, use_bn=False),
            #DownsampleBlock(channel_in=128, channel_out=256, activation_fn=activation, use_bn=False),
        )
        self.deepest_shape = (64, 32 // 4, 32 // 4)

        # FC network
        self.fc = nn.Sequential(
            nn.Linear(in_features=reduce(mul, self.deepest_shape), out_features=64),
            # nn.BatchNorm1d(num_features=256),
            activation,
            nn.Linear(in_features=64, out_features=flatten_size),
            nn.Tanh() # TODO replace with tanh cause our data are scaled to [-1, 1]
        )

    def forward(self, x):
        h = x
        h = self.conv(h)
        h = h.view(len(h), -1)
        o = self.fc(h)
        return o

class _Decoder_LSA(nn.Module):
    """
    CIFAR10 model decoder.
    """
    def __init__(self, in_channels,out_channels, h_shape, hidden_layer_sizes, activation,pad_mode,bn, group=1,
                 last_channel=16, add_conv=False):
        super(_Decoder_LSA, self).__init__()

        self.deepest_shape = (64, 32 // 4, 32 // 4)

        # FC network
        self.fc = nn.Sequential(
            nn.Linear(in_features=in_channels, out_features=64),
            #nn.BatchNorm1d(num_features=256),
            activation,
            nn.Linear(in_features=64, out_features=reduce(mul, self.deepest_shape)),
            activation
        )

        # Convolutional network
        self.conv = nn.Sequential(
            UpsampleBlock(channel_in=64, channel_out=32, activation_fn=activation, use_bn=False),
            UpsampleBlock(channel_in=32, channel_out=16, activation_fn=activation, use_bn=False),
            #UpsampleBlock(channel_in=64, channel_out=32, activation_fn=activation, use_bn=False),
            #ResidualBlock(channel_in=32, channel_out=32, activation_fn=activation, use_bn=False),
            nn.Conv2d(in_channels=16, out_channels=3, kernel_size=1, bias=False)
        )

    def forward(self, x):
        # types: (torch.Tensor) -> torch.Tensor
        """
        Forward propagation.

        :param x: the batch of latent vectors.
        :return: the batch of reconstructions.
        """
        h = x
        h = self.fc(h)
        h = h.view(len(h), *self.deepest_shape)
        h = self.conv(h)
        o = h

        return o

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
            layers.append(nn.Tanh())  # self.activation TODO
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
            layers.append(nn.Tanh()) #TODO
        self.layers = nn.Sequential(*layers)
        print(self.layers)

    def forward(self, x):
        h = self.fc(x)
        h = self.activation(h)
        h = h.view(-1, *self.h_shape)
        if self.bnlayer is not None:
            h = self.bn(h)
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

class SN_MNIST(nn.Module):
    def __init__(self, input_shape, hidden_layer_sizes, z_channels, add_conv):
        super(SN_MNIST, self).__init__()
        c, h, w = input_shape
        pad_mode = 1 if h % 8 == 0 else 0
        self.encoder = _Encoder(in_channels=c, hidden_layer_sizes=hidden_layer_sizes, activation=nn.Tanh(), pad_mode=pad_mode)

        h_shape = (hidden_layer_sizes[2], h//8, w//8)
        h_channels = h_shape[0]*h_shape[1]*h_shape[2]

        self.A = torch.nn.Parameter(torch.randn(h_channels, z_channels))
        #self.N = torch.nn.Parameter(torch.randn(h_channels, z_channels))

        self.decoder = _Decoder(in_channels=z_channels, out_channels=c, h_shape=h_shape,
                                hidden_layer_sizes=hidden_layer_sizes,activation=nn.Tanh(),
                                pad_mode=pad_mode, add_conv=add_conv)

    def forward(self, x):
        y = self.encoder(x)
        y = y.view(y.shape[0], -1)
        y_rsr = torch.matmul(y, self.A)
        #y_rsr = torch.matmul(y_rsr, self.A.t())
        #n_rsr = torch.matmul(y, self.N)
        #z = torch.cat((y_rsr, n_rsr), dim=1)
        z = y_rsr
        x_r = self.decoder(z)
        #print(y.shape, y_rsr.shape, z.shape, x_r.shape)
        return y, y_rsr, z, x_r

class RSRBoneV2Linear(nn.Module):
    def __init__(self, input_shape, hidden_layer_sizes, z_channels, group, bn):
        super(RSRBoneV2Linear, self).__init__()
        c= input_shape
        self.input_shape = input_shape
        self.encoder = _Encoder_Linear(in_channels=c, hidden_layer_sizes=hidden_layer_sizes,
                                       activation=nn.ReLU(), bn=bn)

        self.A = torch.nn.Parameter(torch.randn(hidden_layer_sizes[2], z_channels))
        self.group = group
        self.input_c = c

        self.decoder = _Decoder_Linear(in_channels=z_channels, out_channels=c,
                                hidden_layer_sizes=hidden_layer_sizes,activation=nn.ReLU(),
                                       bn=bn)

    def forward(self, x):
        y = self.encoder(x)
        y_rsr = torch.matmul(y, self.A)
        x_r1 = self.decoder(y_rsr)
        x_r = (x_r1,)
        return y_rsr, x_r, x_r1

class RSRBoneV3Linear(nn.Module):
    def __init__(self, input_shape, hidden_layer_sizes, z_channels, bn):
        super(RSRBoneV3Linear, self).__init__()
        c= input_shape
        self.input_shape = input_shape
        self.encoder = _Encoder_Linear(in_channels=c, hidden_layer_sizes=hidden_layer_sizes,
                                       activation=nn.ReLU(), bn=bn)

        self.A = nn.Linear(in_features=hidden_layer_sizes[2], out_features=z_channels, bias=False)
        #self.A = torch.nn.Parameter(torch.randn(hidden_layer_sizes[2], z_channels))

        self.input_c = c

        self.decoder = _Decoder_Linear(in_channels=z_channels, out_channels=c,
                                hidden_layer_sizes=hidden_layer_sizes,activation=nn.ReLU(),
                                       bn=bn)

        self.estimator = Estimator1D(
            code_length=z_channels,
            fm_list=[32, 32, 32, 32],
            cpd_channels=100
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.encoder(x)
        y_rsr = self.A(y)
        #y_rsr = torch.matmul(y, self.A)
        y_rsr = self.sigmoid(y_rsr)
        x_r = self.decoder(y_rsr)
        z_dist = self.estimator(y_rsr)
        return y_rsr, x_r, z_dist

class RSRBoneV2(nn.Module):
    def __init__(self, input_shape, hidden_layer_sizes, z_channels, group, add_conv):
        super(RSRBoneV2, self).__init__()
        c, h, w = input_shape
        self.input_shape = input_shape
        pad_mode = 1 if h % 8 == 0 else 0
        self.encoder = _Encoder(in_channels=c, hidden_layer_sizes=hidden_layer_sizes, activation=nn.Tanh(),
                                pad_mode=pad_mode)

        h_shape = (hidden_layer_sizes[2], h//8, w//8)
        h_channels = h_shape[0]*h_shape[1]*h_shape[2]
        self.A = torch.nn.Parameter(torch.randn(h_channels, z_channels))
        self.group = group
        self.input_c = c
        last_channel = 16 if add_conv else 1

        self.decoder = _Decoder(in_channels=z_channels, out_channels=c, h_shape=h_shape,
                                hidden_layer_sizes=hidden_layer_sizes,activation=nn.Tanh(),
                                group=group, last_channel=last_channel,pad_mode=pad_mode, add_conv=add_conv)

    def forward(self, x):
        y = self.encoder(x)
        y = y.view(y.shape[0], -1)
        #print(y.shape)
        y_rsr = torch.matmul(y, self.A)
        x_r1 = self.decoder(y_rsr)
        #print(x_r1.shape)
        #x_r1 = x_r1.view(-1, self.group*self.input_shape[0], *self.input_shape[1:])
        #print(x_r1.shape)
        if self.group == 1:
            x_r = (x_r1,)
            return y_rsr, x_r,  x_r1
        elif self.input_c == 1:
            x_r = torch.chunk(x_r1, self.group, dim=1)
            return y_rsr, x_r, x_r1.mean(dim=1, keepdim=True)
        else:
            raise NotImplementedError

class RSRBoneType(nn.Module):
    def __init__(self, input_shape, hidden_layer_sizes, z_channels, mode):
        super(RSRBoneType, self).__init__()
        c, h, w = input_shape
        self.mode = mode

        self.encoder = _Encoder(in_channels=c, hidden_layer_sizes=hidden_layer_sizes, activation=nn.Tanh())

        h_shape = (hidden_layer_sizes[2], h//8, w//8)
        h_channels = h_shape[0]*h_shape[1]*h_shape[2]
        self.A = torch.nn.Parameter(torch.randn(h_channels, z_channels))

        self.decoder = _Decoder(in_channels=h_channels, out_channels=c, h_shape=h_shape,
                                hidden_layer_sizes=hidden_layer_sizes,activation=nn.Tanh())

    def forward(self, x):
        y = self.encoder(x)
        y = y.view(y.shape[0], -1)
        y_rsr = torch.matmul(y, self.A)
        y_rsr = torch.matmul(y_rsr, self.A.t())
        if self.mode == 'A':
            x_r = self.decoder(y_rsr)
        else:
            x_r = self.decoder(y)
        return y, y_rsr, x_r

    def predict(self, x, mod=None):
        if mod is None:
            mod = self.mode
        y = self.encoder(x)
        y = y.view(y.shape[0], -1)
        y_rsr = torch.matmul(y, self.A)
        y_rsr = torch.matmul(y_rsr, self.A.t())
        if mod == 'A':
            x_r = self.decoder(y)
        else:
            x_r = self.decoder(y_rsr)
        return y, y_rsr, x_r

class RSRBoneTypeV2(nn.Module):
    def __init__(self, input_shape, hidden_layer_sizes, z_channels, mode):
        super(RSRBoneTypeV2, self).__init__()
        c, h, w = input_shape
        self.mode = mode
        pad_mode = 1 if h % 8 == 0 else 0
        self.encoder = _Encoder(in_channels=c, hidden_layer_sizes=hidden_layer_sizes, activation=nn.Tanh(),
                                pad_mode=pad_mode)

        h_shape = (hidden_layer_sizes[2], h//8, w//8)
        h_channels = h_shape[0]*h_shape[1]*h_shape[2]
        self.A = torch.nn.Parameter(torch.randn(h_channels, z_channels))
        self.B = torch.nn.Parameter(torch.randn(h_channels, z_channels))

        self.decoder = _Decoder(in_channels=h_channels, out_channels=c, h_shape=h_shape,
                                hidden_layer_sizes=hidden_layer_sizes,activation=nn.Tanh(), pad_mode=pad_mode)

    def forward(self, x):
        y = self.encoder(x)
        y = y.view(y.shape[0], -1)
        z = torch.matmul(y, self.A)
        y_rsr = torch.matmul(z, self.B.t())
        if self.mode == 'A':
            x_r = self.decoder(y_rsr)
        else:
            x_r = self.decoder(y)
        return y, y_rsr, x_r

    def predict(self, x, mod=None):
        if mod is None:
            mod = self.mode
        y = self.encoder(x)
        y = y.view(y.shape[0], -1)
        z = torch.matmul(y, self.A)
        y_rsr = torch.matmul(z, self.B.t())
        if mod == 'A':
            x_r = self.decoder(y)
        else:
            x_r = self.decoder(y_rsr)
        return y, y_rsr, x_r

class RSRBoneTypeV3Linear(nn.Module):
    def __init__(self, input_shape, hidden_layer_sizes, z_channels, mode, bn, shareAB):
        super(RSRBoneTypeV3Linear, self).__init__()
        c = input_shape
        self.mode = mode # A or B or AB
        self.encoder = _Encoder_Linear(in_channels=c, hidden_layer_sizes=hidden_layer_sizes,
                                       activation=nn.ReLU(), bn=bn)


        self.A = torch.nn.Parameter(torch.randn(hidden_layer_sizes[2], z_channels))
        if shareAB:
            self.B = self.A
            print('shareAB')
        else:
            self.B = torch.nn.Parameter(torch.randn(hidden_layer_sizes[2], z_channels))

        self.decoder = _Decoder_Linear(in_channels=hidden_layer_sizes[2], out_channels=c,
                                       hidden_layer_sizes=hidden_layer_sizes, activation=nn.ReLU(),
                                       bn=bn)

    def forward(self, x):
        y = self.encoder(x)
        z = torch.matmul(y, self.A)
        y_rsr = torch.matmul(z, self.B.t())
        if self.mode == 'A':
            x_r = self.decoder(y_rsr)
        elif self.mode == 'B':
            x_r = self.decoder(y)
        else:
            x_r = self.decoder((y_rsr + y)/2.)
        return y, y_rsr, x_r

    def predict(self, x):
        y = self.encoder(x)
        z = torch.matmul(y, self.A)
        y_rsr = torch.matmul(z, self.B.t())
        x_rA = self.decoder(y_rsr)
        x_rB = self.decoder(y)
        x_rAB = self.decoder((y_rsr + y) / 2.)
        return y, y_rsr, (x_rA, x_rB, x_rAB)

class RSRBoneTypeV3(nn.Module):
    def __init__(self, input_shape, hidden_layer_sizes, z_channels, mode, bn, shareAB):
        super(RSRBoneTypeV3, self).__init__()
        c, h, w = input_shape
        pad_mode = 1 if h % 8 == 0 else 0

        self.mode = mode # A or B or AB
        self.encoder = _Encoder(in_channels=c, hidden_layer_sizes=hidden_layer_sizes,
                                       activation=nn.Tanh(), pad_mode=pad_mode, bn=bn)

        h_shape = (hidden_layer_sizes[2], h // 8, w // 8)
        h_channels = h_shape[0] * h_shape[1] * h_shape[2]

        self.A = torch.nn.Parameter(torch.randn(h_channels, z_channels))
        if shareAB:
            self.B = self.A
            print('shareAB')
        else:
            self.B = torch.nn.Parameter(torch.randn(h_channels, z_channels))

        self.decoder = _Decoder(in_channels=h_channels, out_channels=c,h_shape=h_shape,
                                       hidden_layer_sizes=hidden_layer_sizes, activation=nn.Tanh(),
                                       pad_mode=pad_mode, add_conv=False, bn=bn)

    def forward(self, x):
        y = self.encoder(x)
        y = y.view(y.shape[0], -1)
        z = torch.matmul(y, self.A)
        y_rsr = torch.matmul(z, self.B.t())
        if self.mode == 'A':
            x_r = self.decoder(y_rsr)
        elif self.mode == 'B':
            x_r = self.decoder(y)
        else:
            x_r = self.decoder((y_rsr + y)/2.)
        return y, y_rsr, x_r

    def predict(self, x):
        y = self.encoder(x)
        y = y.view(y.shape[0], -1)
        z = torch.matmul(y, self.A)
        y_rsr = torch.matmul(z, self.B.t())
        x_rA = self.decoder(y_rsr)
        x_rB = self.decoder(y)
        x_rAB = self.decoder((y_rsr + y) / 2.)
        return y, y_rsr, (x_rA, x_rB, x_rAB)


class RSRBoneTypeV4Linear(nn.Module):
    def __init__(self, input_shape, hidden_layer_sizes, z_channels, mode, bn):
        super(RSRBoneTypeV4Linear, self).__init__()
        c = input_shape
        self.mode = mode # A or B or AB
        self.encoder = _Encoder_Linear(in_channels=c, hidden_layer_sizes=hidden_layer_sizes,
                                       activation=nn.ReLU(), bn=bn)


        #self.A = torch.nn.Parameter(torch.randn(hidden_layer_sizes[2], z_channels))

        self.inner_net = RSRBoneTypeV3Linear(hidden_layer_sizes[2], hidden_layer_sizes, z_channels, mode='A'
                                             , bn=bn, shareAB=True)

        self.decoder = _Decoder_Linear(in_channels=hidden_layer_sizes[2], out_channels=c,
                                       hidden_layer_sizes=hidden_layer_sizes, activation=nn.ReLU(),
                                       bn=bn)

    def forward(self, x):
        y = self.encoder(x)
        #z = torch.matmul(y, self.A)
        #y_rsr = torch.matmul(z, self.B.t())
        _, _, y_rsr = self.inner_net(y)
        if self.mode == 'A':
            x_r = self.decoder(y_rsr)
        elif self.mode == 'B':
            x_r = self.decoder(y)
        else:
            x_r = self.decoder((y_rsr + y)/2.)
        return y, y_rsr, x_r

    def predict(self, x):
        y = self.encoder(x)
        #z = torch.matmul(y, self.A)
        #y_rsr = torch.matmul(z, self.B.t())
        _, _, y_rsr = self.inner_net(y)
        #print(y.shape, y_rsr.shape)
        x_rA = self.decoder(y_rsr)
        x_rB = self.decoder(y)
        x_rAB = self.decoder((y_rsr + y) / 2.)
        return y, y_rsr, (x_rA, x_rB, x_rAB)

class RSRBoneTypeV5Linear(nn.Module):
    def __init__(self, input_shape, hidden_layer_sizes, z_channels, mode, bn, shareAB):
        super(RSRBoneTypeV5Linear, self).__init__()
        c = input_shape
        self.mode = mode # A or B or AB
        self.encoder = _Encoder_Linear(in_channels=c, hidden_layer_sizes=hidden_layer_sizes,
                                       activation=nn.ReLU(), bn=bn)


        self.A = torch.nn.Parameter(torch.randn(hidden_layer_sizes[2], z_channels))
        if shareAB:
            self.B = self.A
            print('shareAB')
        else:
            self.B = torch.nn.Parameter(torch.randn(hidden_layer_sizes[2], z_channels))

        self.decoder = _Decoder_Linear(in_channels=hidden_layer_sizes[2], out_channels=c,
                                       hidden_layer_sizes=hidden_layer_sizes, activation=nn.ReLU(),
                                       bn=bn)

    def forward(self, x):
        y = self.encoder(x)
        z = torch.matmul(y, self.A)
        y_rsr = torch.matmul(z, self.B.t())
        if self.mode == 'A':
            x_r = self.decoder(y_rsr)
        elif self.mode == 'B':
            x_r = self.decoder(y)
        else:
            x_r = self.decoder((y_rsr + y)/2.)
        return y, y_rsr, x_r

    def predict(self, x):
        y = self.encoder(x)
        z = torch.matmul(y, self.A)
        y_rsr = torch.matmul(z, self.B.t())
        x_rA = self.decoder(y_rsr)
        x_rB = self.decoder(y)
        x_rAB = self.decoder((y_rsr + y) / 2.)
        return y, y_rsr, (x_rA, x_rB, x_rAB)

class RSRBoneTypeV6(nn.Module):
    def __init__(self, input_shape, hidden_layer_sizes, z_channels, mode, bn, shareAB, flatten_size=128):
        super(RSRBoneTypeV6, self).__init__()
        c, h, w = input_shape
        pad_mode = 1 if h % 8 == 0 else 0

        self.mode = mode # A or B or AB
        # flatten_size = 128
        self.encoder = _Encoder(in_channels=c, hidden_layer_sizes=hidden_layer_sizes,
                                       activation=nn.Tanh(), pad_mode=pad_mode, bn=bn, is_flatten=True, #nn.Tanh()
                                flatten_size=flatten_size)

        h_shape = (hidden_layer_sizes[2], h // 8, w // 8)
        h_channels = flatten_size

        self.A = torch.nn.Parameter(torch.randn(h_channels, z_channels))
        if shareAB:
            self.B = self.A
            print('shareAB')
        else:
            self.B = torch.nn.Parameter(torch.randn(h_channels, z_channels))

        self.decoder = _Decoder(in_channels=h_channels, out_channels=c,h_shape=h_shape,
                                       hidden_layer_sizes=hidden_layer_sizes, activation=nn.Tanh(),
                                       pad_mode=pad_mode, add_conv=False, bn=bn)

    def forward(self, x):
        y = self.encoder(x)
        y = y.view(y.shape[0], -1)
        z = torch.matmul(y, self.A)
        y_rsr = torch.matmul(z, self.B.t())
        if self.mode == 'A':
            x_r = self.decoder(y_rsr)
        elif self.mode == 'B':
            x_r = self.decoder(y)
        else:
            x_r = self.decoder((y_rsr + y)/2.)
        return y, y_rsr, x_r

    def predict(self, x):
        y = self.encoder(x)
        y = y.view(y.shape[0], -1)
        z = torch.matmul(y, self.A)
        y_rsr = torch.matmul(z, self.B.t())
        x_rA = self.decoder(y_rsr)
        x_rB = self.decoder(y)
        x_rAB = self.decoder((y_rsr + y) / 2.)
        return y, y_rsr, (x_rA, x_rB, x_rAB)

class RSRBoneTypeV6Linear(nn.Module):
    def __init__(self, input_shape, hidden_layer_sizes, z_channels, mode, bn, shareAB, flatten_size=128):
        super(RSRBoneTypeV6Linear, self).__init__()
        c = input_shape
        self.mode = mode # A or B or AB
        self.encoder = _Encoder_Linear(in_channels=c, hidden_layer_sizes=hidden_layer_sizes,
                                       activation=nn.ReLU(), bn=bn, is_flatten=True, flatten_size=flatten_size)

        h_channels = flatten_size

        self.A = torch.nn.Parameter(torch.randn(h_channels, z_channels))
        if shareAB:
            self.B = self.A
            print('shareAB')
        else:
            self.B = torch.nn.Parameter(torch.randn(h_channels, z_channels))

        self.decoder = _Decoder_Linear(in_channels=h_channels, out_channels=c,
                                       hidden_layer_sizes=hidden_layer_sizes, activation=nn.ReLU(),
                                       bn=bn)

    def forward(self, x):
        y = self.encoder(x)
        z = torch.matmul(y, self.A)
        y_rsr = torch.matmul(z, self.B.t())
        if self.mode == 'A':
            x_r = self.decoder(y_rsr)
        elif self.mode == 'B':
            x_r = self.decoder(y)
        else:
            x_r = self.decoder((y_rsr + y)/2.)
        return y, y_rsr, x_r

    def predict(self, x):
        y = self.encoder(x)
        z = torch.matmul(y, self.A)
        y_rsr = torch.matmul(z, self.B.t())
        x_rA = self.decoder(y_rsr)
        x_rB = self.decoder(y)
        x_rAB = self.decoder((y_rsr + y) / 2.)
        return y, y_rsr, (x_rA, x_rB, x_rAB)


class RSRBoneTypeV7(nn.Module):
    def __init__(self, input_shape, hidden_layer_sizes, z_channels, mode, bn, shareAB, flatten_size=128):
        super(RSRBoneTypeV7, self).__init__()
        c, h, w = input_shape
        pad_mode = 1 if h % 8 == 0 else 0

        self.mode = mode # A or B or AB
        # flatten_size = 128
        self.encoder = _Encoder_LSA(in_channels=c, hidden_layer_sizes=hidden_layer_sizes,
                                       activation=nn.Tanh(), pad_mode=pad_mode, bn=bn, is_flatten=True,
                                flatten_size=flatten_size)

        h_shape = (hidden_layer_sizes[2], h // 8, w // 8)
        h_channels = flatten_size

        self.A = torch.nn.Parameter(torch.randn(h_channels, z_channels))
        if shareAB:
            self.B = self.A
            print('shareAB')
        else:
            self.B = torch.nn.Parameter(torch.randn(h_channels, z_channels))

        self.decoder = _Decoder_LSA(in_channels=h_channels, out_channels=c,h_shape=h_shape,
                                       hidden_layer_sizes=hidden_layer_sizes, activation=nn.Tanh(),
                                       pad_mode=pad_mode, add_conv=False, bn=bn)

    def forward(self, x):
        y = self.encoder(x)
        y = y.view(y.shape[0], -1)
        z = torch.matmul(y, self.A)
        y_rsr = torch.matmul(z, self.B.t())
        if self.mode == 'A':
            x_r = self.decoder(y_rsr)
        elif self.mode == 'B':
            x_r = self.decoder(y)
        else:
            x_r = self.decoder((y_rsr + y)/2.)
        return y, y_rsr, x_r

    def predict(self, x):
        y = self.encoder(x)
        y = y.view(y.shape[0], -1)
        z = torch.matmul(y, self.A)
        y_rsr = torch.matmul(z, self.B.t())
        x_rA = self.decoder(y_rsr)
        x_rB = self.decoder(y)
        x_rAB = self.decoder((y_rsr + y) / 2.)
        return y, y_rsr, (x_rA, x_rB, x_rAB)
