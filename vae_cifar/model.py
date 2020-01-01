import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

channel = 3
# 输入图片的大小是(96*96*3)
class VAE(nn.Module):
    def __init__(self,device=torch.device("cpu")):
        super(VAE, self).__init__()
        # encoder部分：
        self.en_conv1 = nn.Conv2d(channel, 64, 3, 2, 1)  # 64 * 16 * 16
        self.en_bn1 = nn.BatchNorm2d(num_features=64)  # 上同
        self.en_conv2 = nn.Conv2d(64, 128, 3, 2, 1)  # 128 * 8 * 8
        self.en_bn2 = nn.BatchNorm2d(num_features=128)
        self.en_conv3 = nn.Conv2d(128, 256, 3, 2, 1)  # 256 * 4 * 4
        self.en_bn3 = nn.BatchNorm2d(num_features=256)
        # self.en_conv4 = nn.Conv2d(256, 512, 3, 2, 1)  # 512 * 6 * 6
        # self.en_bn4 = nn.BatchNorm2d(num_features=512)
        self.en_fc1 = nn.Linear(256 * 4 * 4, 512 * 2)
        self.device = device

        # decoder 部分
        self.de_fc1 = nn.Linear(512, 256 * 4 * 4)
        self.de_bn1 = nn.BatchNorm2d(256)
        self.de_convT1 = nn.ConvTranspose2d(256, 128, 3, 2, 1, 1)  # 128 * 8 * 8
        self.de_bn2 = nn.BatchNorm2d(128)
        self.de_convT2 = nn.ConvTranspose2d(128, 32, 3, 2, 1, 1)  # 128 * 16 *16
        self.de_bn3 = nn.BatchNorm2d(32)
        self.de_convT3 = nn.ConvTranspose2d(32, 16, 3, 2, 1, 1)  # 32 * 32 * 32
        self.de_bn4 = nn.BatchNorm2d(16)
        self.de_convT4 = nn.ConvTranspose2d(16, channel, 3, 1, 1, 0)  # 3 * 32 * 32

    def encoder(self, x):
        # x 3*96*96
        x = F.relu(self.en_bn1(self.en_conv1(x)))  # 64 * 48 *48
        x = F.relu(self.en_bn2(self.en_conv2(x)))  # 128 * 24 *24
        x = F.relu(self.en_bn3(self.en_conv3(x)))  # 256 * 12 *12
        #x = F.relu(self.en_bn4(self.en_conv4(x)))  # 512 * 6 * 6
        x = x.view(-1, 256 * 4 * 4)
        x = self.en_fc1(x)
        return x, x[:, :512], x[:, 512:]

    def reparameter(self, x):
        self.mu = x[:, :512]
        self.logvar = x[:, 512:]
        std = torch.sqrt(torch.exp(self.logvar))
        epsilon = torch.randn(self.mu.shape[0], 512).to(self.device)
        return self.mu + std * epsilon

    def decoder(self, z):
        z = self.de_fc1(z)
        z = self.de_bn1(z.view(-1, 256, 4, 4))
        z = F.relu(self.de_bn2(self.de_convT1(z)))  # 256 *24 *24
        z = F.relu(self.de_bn3(self.de_convT2(z)))  # 128 * 48 *48
        z = F.relu(self.de_bn4(self.de_convT3(z)))  # 32 * 96 *96
        z = torch.sigmoid(self.de_convT4(z))
        return z

    def forward(self, x):
        h, mu, logvar = self.encoder(x)
        z = self.reparameter(h)
        out = self.decoder(z)
        return out, mu, logvar


def loss_function(x_, x, mu, logvar):
    BCE_loss = nn.BCELoss(reduction='sum')
    reconstruction_loss = BCE_loss(x_, x)
    KL_divergence = -0.5 * torch.sum(1 + logvar - torch.exp(logvar) - mu ** 2)
    return reconstruction_loss + KL_divergence

# 对模型的初始化


def weigth_init(m):
    if isinstance(m, nn.Conv2d):
        init.xavier_uniform_(m.weight.data)
        init.constant_(m.bias.data, 0.1)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.01)
        m.bias.data.zero_()
