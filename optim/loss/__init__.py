import torch
from .segmim import SegMIMLoss
from .text_removal import TextRemovalLoss

def build_criterion(args):
    if args.dataset_file == 'texterase':
        return TextRemovalLoss(args).to(torch.device(args.device))
    elif args.dataset_file == 'segmim':
        return SegMIMLoss(args).to(torch.device(args.device))
    raise ValueError(f'criterion {args.dataset_file} not supported')