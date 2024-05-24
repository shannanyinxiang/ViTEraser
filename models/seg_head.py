import torch.nn as nn

class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class SegHead(nn.Module):
    def __init__(self, in_channel, encoder_stride):
        super(SegHead, self).__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channel,
                out_channels=encoder_stride ** 2, kernel_size=1),
            nn.PixelShuffle(encoder_stride)
        )

    def forward(self, feature):
        seg_res = self.decoder(feature)
        return seg_res


def build_seg_head(args):
    seg_head = SegHead(
        in_channel=args.swin_enc_embed_dim * 8,
        encoder_stride=32
    )    
    return seg_head
