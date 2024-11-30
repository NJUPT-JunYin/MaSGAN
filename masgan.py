import torch
import torch.nn as nn
import torch.nn.functional as F
import vit

device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))


class ChannelAttention(nn.Module):
    def __init__(self, channel):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel * 2, channel // 2),
            nn.ReLU(),
            nn.Linear(channel // 2, channel),
            nn.Sigmoid()
        )

        # 参数初始化

    def forward(self, input):
        b, c, _, _ = input.size()
        avg_y = self.avg_pool(input).view(b, c)
        max_y = self.max_pool(input).view(b, c)

        y = torch.cat([avg_y, max_y], dim=1)
        y = self.fc(y).view(b, c, 1, 1)

        return input * y


class spacialAttention(nn.Module):
    def __init__(self):
        super(spacialAttention, self).__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        y = torch.mean(input, dim=1, keepdim=True)
        y = self.conv(y)
        y = self.bn(y)
        y = self.sigmoid(y)
        return y * input


class Encoder1(nn.Module):
    def __init__(self, in_channels, img_size_w, img_size_h, kernel_size1, kernel_size2):
        super(Encoder1, self).__init__()
        self.inchannels = in_channels
        self.img_size_w = img_size_w
        self.img_size_h = img_size_h
        self.sefconv1 = nn.Sequential(nn.Conv2d(in_channels, 2 * in_channels, kernel_size=3, stride=1, padding=1),
                                      nn.BatchNorm2d(2 * in_channels), nn.LeakyReLU(),
                                      ChannelAttention(2 * in_channels), nn.ReLU(),
                                      nn.Conv2d(2 * in_channels, 4 * in_channels, kernel_size=3, stride=2, padding=1),
                                      nn.BatchNorm2d(4 * in_channels), spacialAttention(), nn.LeakyReLU())

        self.sefconv2 = nn.Sequential(nn.Conv2d(4 * in_channels, 8 * in_channels, kernel_size=3, stride=1, padding=1),
                                      nn.BatchNorm2d(8 * in_channels), nn.LeakyReLU(),
                                      ChannelAttention(8 * in_channels), nn.ReLU(),
                                      nn.Conv2d(8 * in_channels, 16 * in_channels, kernel_size=3, stride=2, padding=1),
                                      nn.BatchNorm2d(16 * in_channels), spacialAttention(), nn.LeakyReLU())

        self.sefconv3 = nn.Sequential(nn.Conv2d(16 * in_channels, 16 * in_channels, kernel_size=3, stride=1, padding=1),
                                      nn.BatchNorm2d(16 * in_channels), nn.LeakyReLU(),
                                      ChannelAttention(16 * in_channels), nn.ReLU(),
                                      nn.Conv2d(16 * in_channels, 8 * in_channels, kernel_size=3, stride=2, padding=1),
                                      nn.BatchNorm2d(8 * in_channels), spacialAttention(), nn.LeakyReLU())

        self.sefconv4 = nn.Sequential(nn.Conv2d(8 * in_channels, 4 * in_channels, kernel_size=3, stride=1, padding=1),
                                      nn.BatchNorm2d(4 * in_channels), nn.LeakyReLU(),
                                      ChannelAttention(4 * in_channels), nn.ReLU(),
                                      nn.Conv2d(4 * in_channels, 2 * in_channels, kernel_size=3, stride=2, padding=1),
                                      nn.BatchNorm2d(2 * in_channels), spacialAttention(), nn.LeakyReLU())

        self.sefconv5 = nn.Sequential(nn.Conv2d(2 * in_channels, 1, kernel_size=5, stride=2, padding=2),
                                      nn.BatchNorm2d(1),
                                      nn.LeakyReLU())

    def forward(self, input):
        input = self.sefconv1(input)
        input = self.sefconv2(input)
        input = self.sefconv3(input)
        input = self.sefconv4(input)
        input = self.sefconv5(input)
        return input


class Decoder(nn.Module):
    def __init__(self, in_channels):
        super(Decoder, self).__init__()

        self.sefdeconv1 = nn.Sequential(
            nn.ConvTranspose2d(1, 2 * in_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            ChannelAttention(2 * in_channels), nn.ReLU(),
            nn.BatchNorm2d(2 * in_channels), spacialAttention(), nn.LeakyReLU())
        self.sefdeconv2 = nn.Sequential(
            nn.ConvTranspose2d(2 * in_channels, 4 * in_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(4 * in_channels),
            nn.LeakyReLU(), ChannelAttention(4 * in_channels), nn.ReLU(),
            nn.ConvTranspose2d(4 * in_channels, 8 * in_channels, kernel_size=3, stride=1, padding=1, output_padding=0),
            nn.BatchNorm2d(8 * in_channels), spacialAttention(), nn.LeakyReLU())
        self.sefdeconv3 = nn.Sequential(
            nn.ConvTranspose2d(8 * in_channels, 16 * in_channels, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(16 * in_channels),
            nn.LeakyReLU(), ChannelAttention(16 * in_channels), nn.ReLU(),
            nn.ConvTranspose2d(16 * in_channels, 16 * in_channels, kernel_size=3, stride=1, padding=1,
                               output_padding=0), nn.BatchNorm2d(16 * in_channels), spacialAttention(), nn.LeakyReLU())
        self.sefdeconv4 = nn.Sequential(
            nn.ConvTranspose2d(16 * in_channels, 8 * in_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(8 * in_channels),
            nn.LeakyReLU(), ChannelAttention(8 * in_channels), nn.ReLU(),
            nn.ConvTranspose2d(8 * in_channels, 4 * in_channels, kernel_size=3, stride=1, padding=1, output_padding=0),
            nn.BatchNorm2d(4 * in_channels), spacialAttention(), nn.LeakyReLU())
        self.sefdeconv5 = nn.Sequential(
            nn.ConvTranspose2d(4 * in_channels, 2 * in_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(2 * in_channels),
            nn.ConvTranspose2d(2 * in_channels, 3, kernel_size=3, stride=1, padding=1, output_padding=0),
            nn.BatchNorm2d(3),
            nn.Tanh())

    def forward(self, input):
        input = self.sefdeconv1(input)
        input = self.sefdeconv2(input)
        input = self.sefdeconv3(input)
        input = self.sefdeconv4(input)
        input = self.sefdeconv5(input)
        return input


class Encoder2(nn.Module):
    def __init__(self, in_channels):
        super(Encoder2, self).__init__()
        self.sefconv1 = nn.Sequential(nn.Conv2d(in_channels, 2 * in_channels, kernel_size=3, stride=1, padding=1),
                                      nn.BatchNorm2d(2 * in_channels), nn.LeakyReLU(),
                                      ChannelAttention(2 * in_channels), nn.ReLU(),
                                      nn.Conv2d(2 * in_channels, 4 * in_channels, kernel_size=3, stride=2, padding=1),
                                      nn.BatchNorm2d(4 * in_channels), spacialAttention(), nn.LeakyReLU())
        self.sefconv2 = nn.Sequential(nn.Conv2d(4 * in_channels, 8 * in_channels, kernel_size=3, stride=1, padding=1),
                                      nn.BatchNorm2d(8 * in_channels), nn.LeakyReLU(),
                                      ChannelAttention(8 * in_channels), nn.ReLU(),
                                      nn.Conv2d(8 * in_channels, 16 * in_channels, kernel_size=3, stride=2, padding=1),
                                      nn.BatchNorm2d(16 * in_channels), spacialAttention(), nn.LeakyReLU())
        self.sefconv3 = nn.Sequential(nn.Conv2d(16 * in_channels, 16 * in_channels, kernel_size=3, stride=1, padding=1),
                                      nn.BatchNorm2d(16 * in_channels), nn.LeakyReLU(),
                                      ChannelAttention(16 * in_channels), nn.ReLU(),
                                      nn.Conv2d(16 * in_channels, 8 * in_channels, kernel_size=3, stride=2, padding=1),
                                      nn.BatchNorm2d(8 * in_channels), spacialAttention(), nn.LeakyReLU())
        self.sefconv4 = nn.Sequential(nn.Conv2d(8 * in_channels, 4 * in_channels, kernel_size=3, stride=1, padding=1),
                                      nn.BatchNorm2d(4 * in_channels), spacialAttention(), nn.LeakyReLU(),
                                      ChannelAttention(4 * in_channels), nn.ReLU(),
                                      nn.Conv2d(4 * in_channels, 2 * in_channels, kernel_size=3, stride=2, padding=1),
                                      nn.BatchNorm2d(2 * in_channels), spacialAttention(), nn.LeakyReLU())
        self.sefconv5 = nn.Sequential(nn.Conv2d(2 * in_channels, 1, kernel_size=5, stride=2, padding=2),
                                      nn.BatchNorm2d(1),
                                      nn.LeakyReLU())

    def forward(self, input):
        input = self.sefconv1(input)
        input = self.sefconv2(input)
        input = self.sefconv3(input)
        input = self.sefconv4(input)
        input = self.sefconv5(input)
        return input


class Dis(nn.Module):
    def __init__(self, in_channels):
        super(Dis, self).__init__()
        self.sefconv1 = nn.Sequential(nn.Conv2d(in_channels, 4 * in_channels, kernel_size=3, stride=1, padding=1),
                                      nn.BatchNorm2d(4 * in_channels), nn.ReLU(), ChannelAttention(4 * in_channels),
                                      nn.Conv2d(4 * in_channels, 8 * in_channels, kernel_size=3, stride=2, padding=1),
                                      nn.BatchNorm2d(8 * in_channels), spacialAttention(),
                                      nn.LeakyReLU()
                                      )
        self.sefconv2 = nn.Sequential(nn.Conv2d(8 * in_channels, 16 * in_channels, kernel_size=3, stride=2, padding=1),
                                      nn.BatchNorm2d(16 * in_channels), nn.ReLU()
                                      )
        self.sefconv3 = nn.Sequential(nn.Conv2d(16 * in_channels, 8 * in_channels, kernel_size=3, stride=2, padding=1),
                                      nn.BatchNorm2d(8 * in_channels), nn.ReLU()
                                      )
        self.sefconv4 = nn.Sequential(nn.Conv2d(8 * in_channels, 4 * in_channels, kernel_size=3, stride=2, padding=1),
                                      nn.BatchNorm2d(4 * in_channels), nn.ReLU()
                                      )
        self.sefconv5 = nn.Sequential(nn.Conv2d(4 * in_channels, 1, kernel_size=3, stride=2, padding=1),
                                      nn.BatchNorm2d(1), nn.ReLU()
                                      )
        self.linear = nn.Sequential(nn.Linear(150, 60), nn.Dropout(0.2))
        self.linear1 = nn.Sequential(nn.Linear(60, 1), nn.Dropout(0.2))

    def forward(self, input):
        input = self.sefconv1(input)
        input = self.sefconv2(input)
        input = self.sefconv3(input)
        input = self.sefconv4(input)
        input = self.sefconv5(input)
        input = input.view(input.size(0), -1)
        input = F.relu(self.linear(input))
        input = F.sigmoid(self.linear1(input))
        return input


class NetG(nn.Module):
    def __init__(self, in_channels1, in_channels2, fre_size1, t_size1, fre_size2, t_size2):
        super(NetG, self).__init__()
        self.in_channels1 = in_channels1
        self.in_channels2 = in_channels2
        self.fre_size1 = fre_size1
        self.fre_size2 = fre_size2
        self.t_size1 = t_size1
        self.t_size2 = t_size2
        self.encoder1 = Encoder1(self.in_channels1)
        self.decoder = Decoder(self.in_channels1)
        self.encoder2 = Encoder2(in_channels1)

    def forward(self, x):
        latent_i = self.encoder1(x)



        gen_imag = self.decoder(latent_i)



        latent_o = self.encoder2(gen_imag)

        return gen_imag, latent_i, latent_o