import torch
from .loss import TextRemovalLoss
from .lr_scheduler import linear_decay_lr

def build_criterion(args):
    return TextRemovalLoss(args).to(torch.device(args.device))

def build_optimizer(model, discriminator, args):
    if args.distributed:
        model_without_ddp = model.module 
        discriminator_without_ddp = discriminator.module if not discriminator is None else None 
    else:
        model_without_ddp = model 
        discriminator_without_ddp = discriminator
    
    optimizer = {}
    G_param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if (not "encoder" in n) and p.requires_grad],
         "lr": args.lr},
        {"params": [p for n, p in model_without_ddp.named_parameters() if ("encoder" in n) and p.requires_grad], 
         "lr": args.lr * args.lr_encoder_ratio},
    ]
    optimizer_G = torch.optim.AdamW(G_param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    optimizer['G'] = optimizer_G

    D_param_dicts = [
        {'params': [p for n, p in discriminator_without_ddp.named_parameters() if p.requires_grad],
        'lr': args.lr}
    ]
    optimizer_D = torch.optim.AdamW(D_param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    optimizer['D'] = optimizer_D

    return optimizer

def get_lr_schedule(args):
    return linear_decay_lr(args)