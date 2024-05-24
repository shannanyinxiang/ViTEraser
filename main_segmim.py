import os 
import time
import json
import torch
import random
import datetime
import numpy as np
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.cuda.amp as amp

from pathlib import Path

from models import build_model
from optim import build_criterion, build_optimizer
from optim.lr_scheduler import get_lr_schedule
from engine.train_segmim import train_one_epoch
from datasets import build_dataloader, build_dataset
from utils.checkpointer import Checkpointer
from utils.misc import get_sha, process_args
from utils.dist import init_distributed_mode, get_rank, is_main_process

cudnn.benchmark = True

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

    model = build_model(args)
    print(model)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    checkpointer = Checkpointer(args.distributed, args.eval)

    train_dataset = build_dataset('train', args)
    train_dataloader, train_sampler = build_dataloader(train_dataset, 'train', args)

    criterion = build_criterion(args)
    optimizer = build_optimizer(model, None, args)
    lr_scheduler = get_lr_schedule(args, optimizer, len(train_dataloader))

    if args.segmim_finetune and args.segmim_ft_init_weight_path != '':
        segmim_ft_init_weight = torch.load(args.segmim_ft_init_weight_path, map_location='cpu')
        if args.distributed:
            model.module.load_state_dict(segmim_ft_init_weight['model'], strict=False)
        else:
            model.load_state_dict(segmim_ft_init_weight['model'], strict=False)

    if args.resume != '':
        start_epoch = checkpointer.load(args.resume, model, None, optimizer)
    else:
        start_epoch = 0

    scaler = amp.GradScaler()

    print("Start training")
    start_time = time.time()
    for epoch in range(start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model=model, 
            criterion=criterion, 
            data_loader=train_dataloader, 
            optimizer=optimizer, 
            scaler=scaler,
            epoch=epoch,
            lr_scheduler=lr_scheduler, 
            args=args,
        )

        if (epoch + 1) % args.save_interval == 0:
            checkpoint_path = os.path.join(args.output_dir, 'checkpoints', f'checkpoint{epoch:04}.pth')
            checkpointer.save(
                checkpoint_path=checkpoint_path,
                model=model,
                discriminator=None,
                optimizer=optimizer,
                epoch=epoch,
                args=args
            )

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if is_main_process():
            with open(os.path.join(args.output_dir, 'log.txt'), 'a') as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    import argparse
    from utils.parser import get_args_parser

    parser = argparse.ArgumentParser('Text Erase Pretraining and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
