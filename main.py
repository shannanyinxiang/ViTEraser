import os
import time
import json
import torch
import random
import datetime
import numpy as np
import torch.distributed as dist

from models import build_model
from engine.val import evaluate
from engine.train import train_one_epoch
from utils.checkpointer import Checkpointer
from utils.misc import get_sha, process_args
from datasets import build_dataloader, build_dataset
from optim import build_criterion, build_optimizer, get_lr_schedule
from utils.dist import init_distributed_mode, get_rank, is_main_process

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

def main(args):
    init_distributed_mode(args)
    print("git:\n  {}\n".format(get_sha()))

    args = process_args(args)
    dist.barrier()
    print(args)

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

    train_dataset = build_dataset('train', args)
    train_dataloader, train_sampler = build_dataloader(train_dataset, 'train', args)

    criterion = build_criterion(args)
    optimizer = build_optimizer(model, discriminator, args)
    if args.resume != '':
        start_epoch = checkpointer.load(args.resume, model, discriminator, optimizer)
    else:
        start_epoch = 0
    learning_rate_schedule = get_lr_schedule(args)

    print("Start training")
    start_time = time.time()
    for epoch in range(start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model=model, 
            discriminator=discriminator, 
            criterion=criterion, 
            data_loader=train_dataloader, 
            optimizer=optimizer, 
            epoch=epoch,
            lr_schedule=learning_rate_schedule, 
            args=args,
        )

        if (epoch + 1) % args.save_interval == 0:
            checkpoint_path = os.path.join(args.output_dir, 'checkpoints', f'checkpoint{epoch:04}.pth')
            checkpointer.save(
                checkpoint_path=checkpoint_path,
                model=model,
                discriminator=discriminator,
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
    from pathlib import Path
    from utils.parser import get_args_parser

    parser = argparse.ArgumentParser('ViTEraser training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
