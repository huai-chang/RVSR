import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import SqueezeExcite

def pad_tensor(t, pattern):
    pattern = pattern.view(1, -1, 1, 1)
    t = F.pad(t, (1, 1, 1, 1), 'constant', 0)
    t[:, :, 0:1, :] = pattern
    t[:, :, -1:, :] = pattern
    t[:, :, :, 0:1] = pattern
    t[:, :, :, -1:] = pattern

    return t

class RepConv(nn.Module):
    """ Residual in residual reparameterizable block.
    Using reparameterizable block to replace single 3x3 convolution.

    Args:
        n_feats (int): The number of feature maps.
        ratio (int): Expand ratio.
    """

    def __init__(self, n_feats, ratio=2):
        super(RepConv, self).__init__()
        self.expand_conv = nn.Conv2d(n_feats, n_feats*ratio, 1, 1, 0)
        self.fea_conv = nn.Conv2d(n_feats*ratio, n_feats*ratio, 3, 1, 0)
        self.reduce_conv = nn.Conv2d(n_feats*ratio, n_feats, 1, 1, 0)

        self.expand1_conv = nn.Conv2d(n_feats, n_feats*ratio, 1, 1, 0)
        self.fea1_conv = nn.Conv2d(n_feats*ratio, n_feats*ratio, 3, 1, 0)
        self.reduce1_conv = nn.Conv2d(n_feats*ratio, n_feats, 1, 1, 0)
        
        self.res_conv3x3 = nn.Conv2d(n_feats, n_feats, 3, 1, 1)
        self.res_conv1x1 = nn.Conv2d(n_feats, n_feats, 1)

    def forward(self, x):
        res3 = self.res_conv3x3(x)
        res1 = self.res_conv1x1(x)
        res  = x

        branch1 = self.expand_conv(x)
        b0 = self.expand_conv.bias
        branch1 = pad_tensor(branch1, b0)
        branch1 = self.fea_conv(branch1)
        branch1 = self.reduce_conv(branch1)

        branch2 = self.expand1_conv(x)
        b0 = self.expand1_conv.bias
        branch2 = pad_tensor(branch2, b0)
        branch2 = self.fea1_conv(branch2)
        branch2 = self.reduce1_conv(branch2)

        out = branch1 + branch2 + res + res1 + res3

        return out
    
class FFN(nn.Module):
    def __init__(self, N):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Conv2d(N, N*2, 1),
            nn.GELU(),
            nn.Conv2d(N*2, N, 1),
        )

    def forward(self, x):
        return self.ffn(x) + x
    
class RepViT(nn.Module):
    def __init__(self, N):
        super().__init__()
        self.token_mixer = RepConv(N)
        self.channel_mixer = FFN(N)
        self.attn = SqueezeExcite(N, 0.25)

    def forward(self, x):
        return self.channel_mixer(self.attn(self.token_mixer(x)))


class RVSR(nn.Module):
    def __init__(self, sr_rate=4, N=16):
        super(RVSR, self).__init__()

        self.scale = sr_rate

        self.head = nn.Sequential(nn.Conv2d(3,N,3,1,1))

        self.body = nn.Sequential(RepViT(N),
                                  RepViT(N),
                                  RepViT(N),
                                  RepViT(N),
                                  RepViT(N),
                                  RepViT(N),
                                  RepViT(N),
                                  RepViT(N))

        self.tail = nn.Sequential(RepConv(N),
                                  nn.Conv2d(N,3*sr_rate*sr_rate,1),
                                  nn.PixelShuffle(4))
    
    def forward(self, x):
        head = self.head(x)

        body = self.body(head) + head
        
        h = self.tail(body)

        base = F.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=False)

        out = h + base
        
        return out