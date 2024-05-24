from .viteraser import build as build_viteraser_model
from .segmim import build as build_segmim_model

def build_model(args):
    if args.dataset_file == 'texterase':
        return build_viteraser_model(args)
    elif args.dataset_file == 'segmim':
        return build_segmim_model(args)
    raise ValueError(f'model {args.dataset_file} not supported')