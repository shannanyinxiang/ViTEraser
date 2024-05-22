import copy 
import torch
import numpy as np
import torch.nn as nn 

from models.block import ConvWithActivation

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        cnum = 32
        self.global_dis = nn.Sequential(
            ConvWithActivation('conv', 3, 2*cnum, 4, 2, padding=get_pad(256, 5, 2)),
            ConvWithActivation('conv', 2*cnum, 4*cnum, 4, 2, padding=get_pad(128, 5, 2)),
            ConvWithActivation('conv', 4*cnum, 8*cnum, 4, 2, padding=get_pad(64, 5, 2)),
            ConvWithActivation('conv', 8*cnum, 8*cnum, 4, 2, padding=get_pad(32, 5, 2)),
            ConvWithActivation('conv', 8*cnum, 8*cnum, 4, 2, padding=get_pad(16, 5, 2)),
            ConvWithActivation('conv', 8*cnum, 8*cnum, 4, 2, padding=get_pad(8, 5, 2)),            
        )
        self.local_dis = copy.deepcopy(self.global_dis)

        fusion_layers = [nn.Conv2d(16*cnum, 1, kernel_size=4), ]
        fusion_layers.append(nn.Sigmoid())
        self.fusion = nn.Sequential(*fusion_layers)
    
    def forward(self, input, mask):
        global_feat = self.global_dis(input)
        local_feat = self.local_dis(input * (1 - mask))

        concat_feat = torch.cat([global_feat, local_feat], 1)
        fused_prob = self.fusion(concat_feat)
        fused_prob = fused_prob * 2 - 1

        return fused_prob

def get_pad(in_, ksize, stride, atrous=1):
    out_ = np.ceil(float(in_)/stride)
    return int(((out_ - 1) * stride + atrous*(ksize-1) + 1 - in_)/2)

def build_discriminator(args):
    return Discriminator()