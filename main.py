import torch
import random
import numpy as np
import torch.distributed as dist

from models import build_model
from engine.val import evaluate
from datasets import build_dataloader, build_dataset
from utils.checkpointer import Checkpointer
from utils.misc import get_sha, process_args
from utils.dist import init_distributed_mode, get_rank

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

def main(args):
    init_distributed_mode(args)
    print("git:\n  {}\n".format(get_sha()))

    args = process_args(args)
    dist.barrier()
    print(args)

    # fix the seed for reproducibility
    seed = args.seed + get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, discriminator = build_model(args)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(model)
    print('number of params:', n_parameters)

    checkpointer = Checkpointer(args.distributed, args.eval)

    if args.eval:
        val_dataset = build_dataset('val', args)
        val_dataloader, _ = build_dataloader(val_dataset, 'val', args)
        assert(args.resume != '')
        checkpointer.load(args.resume, model)
        evaluate(model, val_dataloader, args)
        return


if __name__ == '__main__':
    import argparse
    from pathlib import Path
    from utils.parser import get_args_parser

    parser = argparse.ArgumentParser('ViTEraser training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
