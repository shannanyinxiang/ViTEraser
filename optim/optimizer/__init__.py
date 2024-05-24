from .segmim import build_optimizer as build_segmim_optimizer
from .viteraser import build_optimizer as build_viteraser_optimizer

def build_optimizer(model, discriminator, args):
    if args.dataset_file == 'texterase':
        return build_viteraser_optimizer(model, discriminator, args)
    elif args.dataset_file == 'segmim':
        return build_segmim_optimizer(model, args)
    raise ValueError(f'optimizer {args.dataset_file} not supported')