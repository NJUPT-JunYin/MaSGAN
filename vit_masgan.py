import torch
import torch.nn as nn
import torch.nn.functional as F
import vit

device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
class timeAttention(nn.Module):
    def __init__(self, time_dim):
        super(timeAttention, self).__init__()
        self.adv = nn.AdaptiveAvgPool2d(1)  
        self.fc = nn.Sequential(
            nn.Linear(time_dim, time_dim // 2),
            nn.ReLU(),
            nn.Linear(time_dim // 2, time_dim),
            nn.Sigmoid()
        )

    def forward(self, input):
        b, _, _, t = input.size()
        input1 = input.permute(0, 3, 2, 1)
        y = self.adv(input1).view(b, t)
        y = self.fc(y).view(b, 1, 1, t)
        return input * y


class llstm(nn.Module):
    def __init__(self,h_size,channel,fre_size,t_size):
        super(llstm, self).__init__()
        self.channel=channel
        self.t_size=t_size
        self.f_size=fre_size
        self.linear=nn.Linear(h_size,channel*fre_size).to(device=device)
        self.h_size=h_size
        self.linear2=nn.Linear(t_size,t_size).to(device=device)
        self.dp=nn.Dropout(0.2)
    def lstm(self,input):
        bs, c, i_size, T = input.shape
        input=input.reshape(bs,c*i_size,T)
        _,i_size,_=input.shape
        b_ih=nn.Parameter(torch.rand(4*self.h_size).to(device=device))
        b_hh=nn.Parameter(torch.rand(4*self.h_size).to(device=device))
        b_ch=nn.Parameter(torch.rand(4*self.h_size).to(device=device))
        w_ih=nn.Parameter(torch.rand(4*self.h_size,i_size).to(device=device))
        w_hh=nn.Parameter(torch.rand(4*self.h_size,self.h_size).to(device=device))
        w_ch=nn.Parameter(torch.rand(4*self.h_size,self.h_size).to(device=device))
        prev_h = torch.rand(bs,self.h_size).to(device=device)
        prev_c = torch.rand(bs,self.h_size).to(device=device)
        batch_w_ih = w_ih.unsqueeze(0).tile(bs, 1, 1)
        batch_w_hh = w_hh.unsqueeze(0).tile(bs, 1, 1)
        batch_w_ch = w_ch.unsqueeze(0).tile(bs, 1, 1)
        output_size = self.h_size
        output = torch.zeros(bs, output_size, T).to(device=device)
        for t in range(T):
            x = input[:, :, t]
            w_times_x = torch.bmm(batch_w_ih, x.unsqueeze(-1))
            w_times_x = w_times_x.squeeze(-1)

            w_times_h_prev = torch.bmm(batch_w_hh, prev_h.unsqueeze(-1))
            w_times_h_prev = w_times_h_prev.squeeze(-1)

            w_times_c_prev = torch.bmm(batch_w_ch, prev_c.unsqueeze(-1))
            w_times_c_prev = w_times_c_prev.squeeze(-1)

            i_t = torch.sigmoid( w_times_x[:, :self.h_size] + w_times_c_prev[:, :self.h_size] + w_times_h_prev[:, :self.h_size] + b_ch[:self.h_size]
                + b_hh[:self.h_size] + b_ih[:self.h_size])
            f_t = torch.sigmoid(w_times_x[:, self.h_size:2 * self.h_size] + w_times_c_prev[:, self.h_size:2 * self.h_size] + w_times_h_prev[:,self.h_size:2 * self.h_size] + b_ch[self.h_size:2 * self.h_size]
                + b_hh[self.h_size:2 * self.h_size] + b_ih[self.h_size:2 * self.h_size])
            g_t = torch.tanh(w_times_x[:, 2 * self.h_size:3 * self.h_size] + w_times_c_prev[:, 2 * self.h_size:3 * self.h_size] + w_times_h_prev[:,2 * self.h_size:3 * self.h_size] + b_ch[2 * self.h_size:3 * self.h_size]
                + b_hh[2 * self.h_size:3 * self.h_size] + b_ih[2 * self.h_size:3 * self.h_size])
            o_t = torch.sigmoid(w_times_x[:, 3 * self.h_size:4 * self.h_size] + w_times_c_prev[:, 3 * self.h_size:4 * self.h_size] + w_times_h_prev[:,3 * self.h_size:4 * self.h_size] + b_ch[3 * self.h_size:4 * self.h_size]
                + b_hh[3 * self.h_size:4 * self.h_size] + b_ih[3 * self.h_size:4 * self.h_size])
            prev_c = f_t * prev_c + i_t * g_t
            prev_h = o_t * torch.tanh(prev_c)
            output[:, :, t] = prev_h
        return output
    def forward(self,input):
        output1=self.lstm(input)
        output1=self.dp(F.relu(self.linear(output1.transpose(1,2))))
        output1 = output1.transpose(1, 2)
        output1=output1.reshape(-1,self.channel,self.f_size,self.t_size)+input
        output1=self.dp(F.relu(self.linear2(output1)))
        return output1


'''class ChannelAttention(nn.Module):
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



    def forward(self, input):
        b, c, _, _ = input.size()
        avg_y = self.avg_pool(input).view(b, c)
        max_y = self.max_pool(input).view(b, c)

        y = torch.cat([avg_y, max_y], dim=1)
        y = self.fc(y).view(b, c, 1, 1)

        return input * y'''


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
                                      timeAttention(2 * in_channels), nn.ReLU(),
                                      nn.Conv2d(2 * in_channels, 4 * in_channels, kernel_size=3, stride=2, padding=1),
                                      nn.BatchNorm2d(4 * in_channels), spacialAttention(), nn.LeakyReLU())
        self.block1 = vit.Vit(embed_dim=kernel_size1 * 4 * in_channels, num_heads=4, act_layer=nn.GELU, mlp_ratio=1,
                              in_channels=4 * in_channels,
                              kernel_size=(kernel_size1, 1), depth=1, img_size_w=img_size_w, img_size_h=img_size_h,
                              stride=(kernel_size1, 1))
        self.sefconv2 = nn.Sequential(nn.Conv2d(4 * in_channels, 8 * in_channels, kernel_size=3, stride=1, padding=1),
                                      nn.BatchNorm2d(8 * in_channels), nn.LeakyReLU(),
                                      timeAttention(8 * in_channels), nn.ReLU(),
                                      nn.Conv2d(8 * in_channels, 16 * in_channels, kernel_size=3, stride=2, padding=1),
                                      nn.BatchNorm2d(16 * in_channels), spacialAttention(), nn.LeakyReLU())
        self.block2 = vit.Vit(embed_dim=kernel_size1 * 16 * in_channels, num_heads=4, act_layer=nn.GELU, mlp_ratio=1,
                              in_channels=16 * in_channels,
                              kernel_size=(kernel_size1, 1), depth=1, img_size_w=img_size_w // 2,
                              img_size_h=img_size_h // 2,
                              stride=(kernel_size1, 1))
        self.sefconv3 = nn.Sequential(nn.Conv2d(16 * in_channels, 16 * in_channels, kernel_size=3, stride=1, padding=1),
                                      nn.BatchNorm2d(16 * in_channels), nn.LeakyReLU(),
                                      timeAttention(16 * in_channels), nn.ReLU(),
                                      nn.Conv2d(16 * in_channels, 8 * in_channels, kernel_size=3, stride=2, padding=1),
                                      nn.BatchNorm2d(8 * in_channels), spacialAttention(), nn.LeakyReLU())
        self.block3 = vit.Vit(embed_dim=kernel_size2 * 8 * in_channels, num_heads=4, act_layer=nn.GELU, mlp_ratio=1,
                              in_channels=8 * in_channels,
                              kernel_size=(kernel_size2, 1), depth=1, img_size_w=img_size_w // 4,
                              img_size_h=img_size_h // 4,
                              stride=(kernel_size2, 1))
        self.sefconv4 = nn.Sequential(nn.Conv2d(8 * in_channels, 4 * in_channels, kernel_size=3, stride=1, padding=1),
                                      nn.BatchNorm2d(4 * in_channels), nn.LeakyReLU(),
                                      timeAttention(4 * in_channels), nn.ReLU(),
                                      nn.Conv2d(4 * in_channels, 2 * in_channels, kernel_size=3, stride=2, padding=1),
                                      nn.BatchNorm2d(2 * in_channels), spacialAttention(), nn.LeakyReLU())
        self.block4 = vit.Vit(embed_dim=kernel_size2 * 2 * in_channels, num_heads=4, act_layer=nn.GELU, mlp_ratio=1,
                              in_channels=2 * in_channels,
                              kernel_size=(kernel_size1, 1), depth=1, img_size_w=img_size_w // 8,
                              img_size_h=img_size_h // 8,
                              stride=(kernel_size1, 1))
        self.sefconv5 = nn.Sequential(nn.Conv2d(2 * in_channels, 1, kernel_size=5, stride=2, padding=2),
                                      nn.BatchNorm2d(1),
                                      nn.LeakyReLU())

    def forward(self, input):
        input = self.block1(self.sefconv1(input)).reshape(-1, 4 * self.inchannels, self.img_size_w // 2,
                                                          self.img_size_h // 2)
        input = self.block2(self.sefconv2(input)).reshape(-1, 16 * self.inchannels, self.img_size_w // 4,
                                                          self.img_size_h // 4)
        input = self.block3(self.sefconv3(input)).reshape(-1, 8 * self.inchannels, self.img_size_w // 8,
                                                          self.img_size_h // 8)
        input = self.block4(self.sefconv4(input)).reshape(-1, 2 * self.inchannels, self.img_size_w // 16,
                                                          self.img_size_h // 16)
        input = self.sefconv5(input)
        return input


class Decoder(nn.Module):
    def __init__(self, in_channels):
        super(Decoder, self).__init__()

        self.sefdeconv1 = nn.Sequential(
            nn.ConvTranspose2d(1, 2 * in_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            timeAttention(2 * in_channels), nn.ReLU(),
            nn.BatchNorm2d(2 * in_channels), spacialAttention(), nn.LeakyReLU())
        self.sefdeconv2 = nn.Sequential(
            nn.ConvTranspose2d(2 * in_channels, 4 * in_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(4 * in_channels),
            nn.LeakyReLU(), timeAttention(4 * in_channels), nn.ReLU(),
            nn.ConvTranspose2d(4 * in_channels, 8 * in_channels, kernel_size=3, stride=1, padding=1, output_padding=0),
            nn.BatchNorm2d(8 * in_channels), spacialAttention(), nn.LeakyReLU())
        self.sefdeconv3 = nn.Sequential(
            nn.ConvTranspose2d(8 * in_channels, 16 * in_channels, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(16 * in_channels),
            nn.LeakyReLU(), timeAttention(16 * in_channels), nn.ReLU(),
            nn.ConvTranspose2d(16 * in_channels, 16 * in_channels, kernel_size=3, stride=1, padding=1,
                               output_padding=0), nn.BatchNorm2d(16 * in_channels), spacialAttention(), nn.LeakyReLU())
        self.sefdeconv4 = nn.Sequential(
            nn.ConvTranspose2d(16 * in_channels, 8 * in_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(8 * in_channels),
            nn.LeakyReLU(), timeAttention(8 * in_channels), nn.ReLU(),
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
                                      timeAttention(2 * in_channels), nn.ReLU(),
                                      nn.Conv2d(2 * in_channels, 4 * in_channels, kernel_size=3, stride=2, padding=1),
                                      nn.BatchNorm2d(4 * in_channels), spacialAttention(), nn.LeakyReLU())
        self.sefconv2 = nn.Sequential(nn.Conv2d(4 * in_channels, 8 * in_channels, kernel_size=3, stride=1, padding=1),
                                      nn.BatchNorm2d(8 * in_channels), nn.LeakyReLU(),
                                      timeAttention(8 * in_channels), nn.ReLU(),
                                      nn.Conv2d(8 * in_channels, 16 * in_channels, kernel_size=3, stride=2, padding=1),
                                      nn.BatchNorm2d(16 * in_channels), spacialAttention(), nn.LeakyReLU())
        self.sefconv3 = nn.Sequential(nn.Conv2d(16 * in_channels, 16 * in_channels, kernel_size=3, stride=1, padding=1),
                                      nn.BatchNorm2d(16 * in_channels), nn.LeakyReLU(),
                                      timeAttention(16 * in_channels), nn.ReLU(),
                                      nn.Conv2d(16 * in_channels, 8 * in_channels, kernel_size=3, stride=2, padding=1),
                                      nn.BatchNorm2d(8 * in_channels), spacialAttention(), nn.LeakyReLU())
        self.sefconv4 = nn.Sequential(nn.Conv2d(8 * in_channels, 4 * in_channels, kernel_size=3, stride=1, padding=1),
                                      nn.BatchNorm2d(4 * in_channels), spacialAttention(), nn.LeakyReLU(),
                                      timeAttention(4 * in_channels), nn.ReLU(),
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
                                      nn.BatchNorm2d(4 * in_channels), nn.ReLU(), timeAttention(4 * in_channels),
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

        llstm1 = nn.Sequential(
            llstm(h_size=64, channel=self.in_channels2, fre_size=self.fre_size1, t_size=self.t_size1), nn.LeakyReLU())
        latent_ii = llstm1(latent_i)

        gen_imag = self.decoder(latent_ii)

        llstm2 = nn.Sequential(
            llstm(h_size=64, channel=self.in_channels1, fre_size=self.fre_size2, t_size=self.t_size2), nn.LeakyReLU())
        lstm_gen_imag = llstm2(gen_imag)

        latent_o = self.encoder2(lstm_gen_imag)

        return gen_imag, latent_i, latent_o
