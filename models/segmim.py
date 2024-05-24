import torch
import torch.nn as nn
import torch.nn.functional as F

from .decoder import build_decoder
from .encoder import build_encoder
from .seg_head import build_seg_head


class ViTEraserSegMIM(nn.Module):
    def __init__(self, encoder, decoder, seg_head):
        super(ViTEraserSegMIM, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        if not decoder is None:
            self.pixel_embed = nn.Linear(encoder.num_channels, decoder.embed_dim)
        self.seg_head = seg_head

    def forward(self, images, masks):

        enc_ms_feats = self.encoder(images, masks)

        preds = {}
        if not self.decoder is None:
            enc_feat = self.pixel_embed(
                    enc_ms_feats[-1].permute(0, 2, 3, 1)
                ).permute(0, 3, 1, 2)
            mim_outputs, _ = self.decoder(enc_feat, enc_ms_feats)
            preds['mim'] = mim_outputs    

        textseg_outputs = self.seg_head(enc_ms_feats[-1])  
        preds['textseg'] = textseg_outputs

        return preds 
    
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"cpb_mlp", "logit_scale", 'relative_position_bias_table'}
        

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


def build(args):
    encoder = build_encoder(args)
    decoder = build_decoder(args) if not args.segmim_finetune else None 
    seg_head = build_seg_head(args)

    model = ViTEraserSegMIM(
        encoder=encoder, 
        decoder=decoder, 
        seg_head=seg_head)

    if args.segmim_finetune:
        for name, param in model.named_parameters():
            if name in ['encoder.backbone.norm0.weight', 'encoder.backbone.norm0.bias', 'encoder.backbone.norm1.weight', 'encoder.backbone.norm1.bias', 'encoder.backbone.norm2.weight', 'encoder.backbone.norm2.bias']:
                param.requires_grad_(False)

    device = torch.device(args.device)
    model = model.to(device)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.gpu],
        )

    return model