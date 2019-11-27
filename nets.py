import torch as t
import torch.nn as nn
import torch.nn.functional as F


#################################
# ## TOY NETWORK FOR 2D DATA ## #
#################################

class ToyNet(nn.Module):
    def __init__(self, dim=2, n_f=32, leak=0.05):
        super(ToyNet, self).__init__()
        self.f = nn.Sequential(
            nn.Conv2d(dim, n_f, 1, 1, 0),
            nn.LeakyReLU(leak),
            nn.Conv2d(n_f, n_f * 2, 1, 1, 0),
            nn.LeakyReLU(leak),
            nn.Conv2d(n_f * 2, n_f * 2, 1, 1, 0),
            nn.LeakyReLU(leak),
            nn.Conv2d(n_f * 2, n_f * 2, 1, 1, 0),
            nn.LeakyReLU(leak),
            nn.Conv2d(n_f * 2, 1, 1, 1, 0))

    def forward(self, x):
        return self.f(x).squeeze()


#########################
# ## VANILLA CONVNET ## #
#########################

class VanillaNet(nn.Module):
    def __init__(self, n_c=3, n_f=32, leak=0.05):
        super(VanillaNet, self).__init__()
        self.f = nn.Sequential(
            nn.Conv2d(n_c, n_f, 3, 1, 1),
            nn.LeakyReLU(leak),
            nn.Conv2d(n_f, n_f * 2, 4, 2, 1),
            nn.LeakyReLU(leak),
            nn.Conv2d(n_f*2, n_f*4, 4, 2, 1),
            nn.LeakyReLU(leak),
            nn.Conv2d(n_f*4, n_f*8, 4, 2, 1),
            nn.LeakyReLU(leak),
            nn.Conv2d(n_f*8, 1, 4, 1, 0))

    def forward(self, x):
        return self.f(x).squeeze()


######################
# ## NONLOCAL NET ## #
######################
# implementation with minor changes from https://github.com/AlexHex7/Non-local_pytorch
# Original Version: Copyright (c) 2018 AlexHex7

class NonlocalNet(nn.Module):
    def __init__(self, n_c=3, n_f=32, leak=0.05):
        super(NonlocalNet, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=n_c, out_channels=n_f, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=leak),
            nn.MaxPool2d(2),

            NonLocalBlock(in_channels=n_f),
            nn.Conv2d(in_channels=n_f, out_channels=n_f * 2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=leak),
            nn.MaxPool2d(2),

            NonLocalBlock(in_channels=n_f * 2),
            nn.Conv2d(in_channels=n_f * 2, out_channels=n_f * 4, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=leak),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(
            nn.Linear(in_features=(n_f * 4) * 4 * 4, out_features=n_f * 8),
            nn.LeakyReLU(negative_slope=leak),
            nn.Linear(in_features=n_f * 8, out_features=1)
        )

    def forward(self, x):
        conv_out = self.convs(x).view(x.shape[0], -1)
        return self.fc(conv_out).squeeze()

# structure of non-local block (from Non-Local Neural Networks https://arxiv.org/abs/1711.07971)
class NonLocalBlock(nn.Module):
    def __init__(self, in_channels, sub_sample=True):
        super(NonLocalBlock, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = max(1, in_channels // 2)

        self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        self.W = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                           kernel_size=1, stride=1, padding=0)
        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)
        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                               kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, nn.MaxPool2d(kernel_size=(2, 2)))
            self.phi = nn.Sequential(self.phi, nn.MaxPool2d(kernel_size=(2, 2)))

    def forward(self, x):

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = t.matmul(theta_x, phi_x)
        f_div_c = F.softmax(f, dim=-1)

        y = t.matmul(f_div_c, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        w_y = self.W(y)
        z = w_y + x

        return z
