import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import SqueezeExcite

class RepConv(nn.Module):
    def __init__(self, N):
        super(RepConv, self).__init__()
        self.rep_conv = nn.Conv2d(N, N, 3, 1, 1)

    def forward(self, x):
        out = self.rep_conv(x)

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
                                  RepViT(N),)

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