'''
Software ExPI
Copyright Inria
Year 2021
Contact : wen.guo@inria.fr
GPL license.
'''
# XIA.py

from torch.nn import Module
from torch import nn
import torch

class XIA_multi(Module):
    def __init__(self, embed_dim=256, nb_h=8, dropout=0.1, nb_att=2):
        super(XIA_multi, self).__init__()

        self.xia_blocs = nn.ModuleList([XIA(embed_dim=embed_dim, nb_h=nb_h, dropout=dropout)
                                    for i in range(nb_att)])

    def forward(self, k1, k2):
        for xia in self.xia_blocs:
            k1 = xia(k1,k2)
        return k1


class XIA(Module):

    def __init__(self, embed_dim=256, nb_h=8, dropout=0.1):
        super(XIA, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, nb_h, dropout=dropout)

        self.fc = nn.Sequential(nn.LayerNorm(embed_dim),
                            nn.Linear(embed_dim,embed_dim),
                            nn.ReLU(),
                            nn.Linear(embed_dim,embed_dim),
                            nn.LayerNorm(embed_dim))

    def forward(self, k1, k2):
        # return k1_new
        query = k2.permute(2,0,1)
        key = k1.permute(2,0,1)
        value = k1.permute(2,0,1)
        k1=k1.permute(2,0,1)

        k = self.self_attn(query, key, value=value)[0]
        k1 = k1+k
        k1 = self.fc(k1)
        return k1.permute(1,2,0)


