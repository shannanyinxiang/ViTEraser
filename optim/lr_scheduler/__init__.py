from .linear_decay import linear_decay_lr
from .multistep import MultiStepLRScheduler
from .cosine import CosineLRScheduler

def get_lr_schedule(args, optimizer=None, n_iter_per_epoch=None):
    if args.dataset_file == 'texterase':
        return linear_decay_lr(args)
    elif args.dataset_file == 'segmim':
        warmup_steps = args.warmup_epochs * n_iter_per_epoch
        num_steps = args.epochs * n_iter_per_epoch 
        if args.segmim_finetune:
            return CosineLRScheduler(
                optimizer,
                t_initial=num_steps - warmup_steps,
                t_mul=1.,
                lr_min=args.min_lr,
                warmup_lr_init=args.warmup_min_lr,
                warmup_t=warmup_steps,
                cycle_limit=1,
                t_in_epochs=False,
                warmup_prefix=True,
            )
        else:
            return MultiStepLRScheduler(
                optimizer, 
                milestones=[i * n_iter_per_epoch for i in args.milestones],
                gamma=0.1,
                warmup_lr_init=args.warmup_min_lr,
                warmup_t=args.warmup_epochs * n_iter_per_epoch,
                t_in_epochs=False
            )