import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .decoder import build_decoder
from .encoder import build_encoder
from .discriminator import build_discriminator
from .vgg16 import build_vgg16

class ViTEraser(nn.Module):
    def __init__(self, encoder, decoder, vgg16, args):
        super(ViTEraser, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.pixel_embed = nn.Linear(encoder.num_channels, decoder.embed_dim)
        self.vgg16 = vgg16

    def forward(self, samples):

        enc_ms_feats = self.encoder(samples)

        enc_feat = self.pixel_embed(
                enc_ms_feats[-1].permute(0, 2, 3, 1)
            ).permute(0, 3, 1, 2)

        outputs, _ = self.decoder(enc_feat, enc_ms_feats)
        return outputs


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def load_pretrained_model(model, weight_path, ignored_backbone=False):
    weight = torch.load(weight_path, map_location='cpu')['model']
    model_dict = model.state_dict()

    loaded_keys = []
    ignore_keys = []
    for k, v in weight.items():
        if 'relative_coords_table' in k or \
               'relative_position_index' in k:
            ignore_keys.append(k)
            continue 

        if ignored_backbone and 'backbone' in k:
            ignore_keys.append(k)
            continue
        
        if k in model_dict.keys():
            model_dict[k] = v
            loaded_keys.append(k)
        else:
            ignore_keys.append(k)
        
    model.load_state_dict(model_dict)
    print(f'Load Model from {weight_path}')
    print('Loaded keys:', loaded_keys)
    print('Ignored keys:', ignore_keys)
    return model


def build(args):
    encoder = build_encoder(args)
    decoder = build_decoder(args)
    discriminator = build_discriminator(args) if not args.eval else None 
    vgg16 = build_vgg16(args) if not args.eval else None 

    model = ViTEraser(
        encoder=encoder, 
        decoder=decoder, 
        vgg16=vgg16, 
        args=args,
    )

    if args.pretrained_model:
        model = load_pretrained_model(model, args.pretrained_model, args.load_pre_ignore_backbone)

    device = torch.device(args.device)
    model = model.to(device)
    if discriminator:
        discriminator = discriminator.to(device)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.gpu],
        )
        
        if discriminator:
            discriminator = torch.nn.parallel.DistributedDataParallel(
                discriminator,
                device_ids=[args.gpu],
            )

    return model, discriminator