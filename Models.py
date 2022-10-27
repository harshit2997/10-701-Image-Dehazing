import torch


class simpleBlock(nn.Module):
    def __init__(self,input_channels,output_channels):
        super().__init__()
        self.conv1 = nn.Conv