import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class baseAE(nn.Module):

    def __init__(self, input, h1_dim, h2_dim, z_dim, class_num):
        super(baseAE, self).__init__()
        self.enc_1 = nn.Linear(input, h1_dim)
        self.enc_2 = nn.Linear(h1_dim, h2_dim)
        self.modelEx = nn.Linear(h2_dim, z_dim)
        self.modelEn = nn.Linear(h2_dim, z_dim)
        self.modelHe = nn.Linear(h2_dim, z_dim)
        self.dec_1 = nn.Linear(z_dim, h2_dim)
        self.dec_2 = nn.Linear(h2_dim, h1_dim)
        self.output = nn.Linear(h1_dim, input)
        self.priorEx = Parameter(torch.Tensor(class_num, z_dim))
        torch.nn.init.xavier_normal_(self.priorEx.data)

    def encode(self, x):
        h1 = F.relu(self.enc_1(x))
        h2 = F.relu(self.enc_2(h1))
        return self.modelEx(h2), self.modelEn(h2)

    def decode(self, z):
        h3 = F.relu(self.dec_1(z))
        h4 = F.relu(self.dec_2(h3))
        return self.output(h4)

    def forward(self, x):
        z, _ = self.encode(x)
        x_reconst = self.decode(z)
        return x_reconst
