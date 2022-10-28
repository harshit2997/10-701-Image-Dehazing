"""
Derived a decent amount of this code from a blog post about the UNet
architecture mentioned here: https://amaarora.github.io/2020/09/13/unet.html

Then got some ideas about how to make this extensible from some of Harshit's old code that
used the UNet architecture based for a different application that can be found here:
https://github.com/thunil/Deep-Flow-Prediction/blob/master/train/DfpNet.py

This file will contain all of the architectures we are going to experiment with throughout this project.
"""


import torch
import torchvision as torchvision
from torch import nn


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, activation='ReLU', kernel_size=(3, 3), bn=False, dropout=0):
        super().__init__()
        # Setup the two convolution layers for the UNet block
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size)

        # Default values for the batching + dropout layers
        self.batch = None
        self.dropout = None

        # Setup the activation function for a single block in our UNet:
        if activation == 'ReLU':
            self.act_func = nn.ReLU()
        else:
            self.act_func = nn.LeakyReLU()

        # Setup the batching layer if it is desired
        if bn:
            self.batch = nn.BatchNorm2d(out_channels)

        # Setup the dropout layer if it is desired
        if dropout > 0:
            self.dropout = nn.Dropout2d(dropout, inplace=True)

    """
    Overridden method for the module to compute a forward pass for a single block
    """
    def forward(self, input_image):
        y = self.conv1(input_image)
        y = self.act_func(y)
        y = self.conv2(y)
        if self.batch is not None:
            y = self.batch(y)
        if self.dropout is not None:
            y = self.dropout(y)
        return self.act_func(y)


class Encoder(nn.Module):
    def __init__(self, chs=(3, 64, 128, 256, 512, 1024), activation='ReLU', kernel_size=(3, 3), bn=False, dropout=0):
        super().__init__()
        enc_list = []
        self.pool = nn.MaxPool2d(2)
        for i in range(len(chs) - 1):
            enc_list.append(Block(chs[i], chs[i + 1], activation, kernel_size, bn, dropout))
        self.encoder_blocks = nn.ModuleList(enc_list)

    """
    Overridden method for the module to compute a forward pass for an encoder
    """
    def forward(self, image):
        enc_features = []
        for block in self.encoder_blocks:
            image = block(image)
            enc_features.append(image)
            image = self.pool(image)
        return enc_features


class Decoder(nn.Module):
    def __init__(self, chs=(1024, 512, 256, 128, 64), activation='ReLU', kernel_size=(3, 3), bn=False, dropout=0):
        super().__init__()
        conv_list = []
        dec_list = []
        for i in range(len(chs) - 1):
            conv_list.append(nn.ConvTranspose2d(chs[i], chs[i + 1], (2, 2), (2, 2)))
            dec_list.append(Block(chs[i], chs[i + 1], activation, kernel_size, bn, dropout))
        self.upsample_list = nn.ModuleList(conv_list)
        self.decoder_blocks = nn.ModuleList(dec_list)

    """
    Overridden method for the module to compute a forward pass for a decoder
    Need to take in the encoder features to use again in the decoder
    """
    def forward(self, image, enc_features):
        for i in range(len(self.upsample_list)):
            # Get the individual operations from the list in the decoder
            upsample_layer = self.upsample_list[i]
            block = self.decoder_blocks[i]
            encoder_output = enc_features[i]
            image = upsample_layer(image)
            _, _, H, W = image.shape

            # Crop the input features and append them to the current decoder output
            cropped_features = torchvision.transforms.CenterCrop([H, W])(encoder_output)
            image = torch.cat([image, cropped_features], dim=1)
            image = block(image)
        return image


class UNet(nn.Module):
    def __init__(self, enc_chs=(3, 64, 128, 256, 512, 1024), dec_chs=(1024, 512, 256, 128, 64), num_class=3,
                 retain_dim=False, out_sz=(572, 572), activation='ReLU', kernel_size=(3, 3), bn=False, dropout=0):
        super().__init__()
        self.encoder = Encoder(enc_chs, activation, kernel_size, bn, dropout)
        self.decoder = Decoder(dec_chs, activation, kernel_size, bn, dropout)
        self.final_layer = nn.Conv2d(dec_chs[-1], num_class, (1, 1))
        self.retain_dim = retain_dim
        self.output_size = out_sz

    """
    Overridden method for the module to compute a forward pass the UNet
    """
    def forward(self, image):
        enc_features = self.encoder(image)
        output = self.decoder(enc_features[::-1][0], enc_features[::-1][1:])
        output = self.final_layer(output)
        if self.retain_dim:
            return nn.functional.interpolate(output, self.output_size)
        else:
            return output


# unet = UNet(retain_dim=True)
# x = torch.randn(1, 3, 572, 572)
# print(unet(x).shape)
