import torch
# torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class NeRF(nn.Module):
    def __int__(self,D=8,W=256,input_ch=3,input_ch_views=3,output_ch=4,skips=[4],use_viewdirs=False):

        super(NeRF,self).__init__()

        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs

        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch,W)+nn.Linear(W,W) if i not in self.skips else nn.Linear(W+input_ch,W) for i in range(D-1)]
        )

        self.views_linears =nn.ModuleList([nn.Linear(input_ch_views + W,W//2)])

        if use_viewdirs:
            self.feature_linear = nn.Linear(W,W)
            self.alpha_linear =nn.Linear(W,1)
            self.rgb_linear = nn.Linear(W // 2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self,x):
        
        return