import torch

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