#from keras.models import Model
#from keras.layers import Input, Conv2D, Activation, BatchNormalization, Flatten, Dense, Conv2DTranspose, Reshape
from utils import get_channels_axis
import torch
import torch.nn as nn
from .encoders_decoders import CAE_pytorch

class AE_pytorch(nn.Module):
    def __init__(self, rep_dim = 256, num_channels=[128, 64, 32]):
        super(AE_pytorch, self).__init__()

        self.num_channels = num_channels
        num_channels = [rep_dim] + num_channels

        # Encoder
        encoder = []
        for in_channels, out_channels in zip(num_channels[:-1], num_channels[1:]):
            block = nn.Sequential(
                nn.Linear(in_channels, out_channels),
                nn.ReLU(inplace=True),
            )
            encoder.append(block)
        self.encoder = nn.Sequential(*encoder)

        # Decoder
        decoder = []
        num_channels = num_channels[::-1][:-1]
        for in_channels, out_channels in zip(num_channels[:-1], num_channels[1:]):
            block = nn.Sequential(
                nn.Linear(in_channels, out_channels),
                nn.ReLU(inplace=True),
            )
            decoder.append(block)
        decoder.append(nn.Sequential(
            nn.Linear(num_channels[-1], rep_dim),
            nn.Tanh()
        ))
        self.decoder = nn.Sequential(*decoder)


    def encode(self, x):
        return self.encoder(x)

    def decode(self, rep):
        return self.decoder(rep)

    def forward(self, x):
        return self.decode(self.encode(x))


class AeInAe_pytorch(nn.Module):
    def __init__(self, in_channels = 3, rep_dim=256):
        super(AeInAe_pytorch, self).__init__()

        self.cae = CAE_pytorch(in_channels, rep_dim)
        self.ae = AE_pytorch(rep_dim)

    def forward(self, x):
        rep = self.cae.encode(x)
        out = self.cae.decode(rep)

        aep = self.ae.encode(rep)
        oaep = self.ae.decode(aep)

        return out, rep, aep, oaep
    

