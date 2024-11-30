import torch
import torch.nn as nn
class MultiScaleConv(nn.Module):
    def __init__(self, in_channels):
        super(MultiScaleConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2)
        self.conv_fuse = nn.Conv2d(in_channels * 3, in_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x3 = self.conv3(x)
        x5 = self.conv5(x)
        x = torch.cat((x1, x3, x5), dim = 1)
        x = self.conv_fuse(x)
        return x
class T(nn.Module):
    def __init__(self,in_channels):
        super(T, self).__init__()
        self.multi_scale_conv = MultiScaleConv(in_channels)

    def forward(self, input):
        input = self.multi_scale_conv(input)
        bs, c, f, t = input.shape
        input_shifted = input.clone()

        for i in range(t - 2):
            channel=i%c
            t1=input_shifted[:, channel, :, i].clone()
            t2=input_shifted[:, channel, :, i+1].clone()
            input_shifted[:, channel, :, i]=t2
            input_shifted[:, channel, :, i + 1]=t1

        return input_shifted