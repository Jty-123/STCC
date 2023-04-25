import torch.nn as nn
import torch
from torch.nn.functional import normalize

class Network_STCC(nn.Module):
    def __init__(self, stcc, feature_dim, class_num):
        super(Network_STCC, self).__init__()
        self.stcc = stcc
        self.feature_dim = feature_dim
        self.cluster_num = class_num
        self.mid_dim = self.stcc.dim * 4
        self.instance_projector = nn.Sequential(
            nn.Linear(self.stcc.dim, self.mid_dim),
            nn.BatchNorm1d(self.mid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.mid_dim, self.mid_dim),
            nn.BatchNorm1d(self.mid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.mid_dim, self.feature_dim),
        )
        self.cluster_projector = nn.Sequential(
            nn.Linear(self.stcc.dim, self.mid_dim),
            nn.BatchNorm1d(self.mid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.mid_dim, self.mid_dim),
            nn.BatchNorm1d(self.mid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.mid_dim, self.cluster_num),
            nn.Softmax(dim=1)
        )

    def forward(self, x_i, x_j):
        print(x_i.shape)
        h_i = self.stcc(x_i)
        h_j = self.stcc(x_j)

        z_i = normalize(self.instance_projector(h_i), dim=1)
        z_j = normalize(self.instance_projector(h_j), dim=1)

        c_i = self.cluster_projector(h_i)
        c_j = self.cluster_projector(h_j)

        return z_i, z_j, c_i, c_j

    def forward_cluster(self, x):
        h = self.stcc(x)
        c = self.cluster_projector(h)
        c = torch.argmax(c, dim=1)
        return c
