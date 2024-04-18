import torch
from .collate_fn import CollateFN
from .texterase import build as build_texterase

def build_dataset(image_set, args):
    if args.dataset_file == 'texterase':
        return build_texterase(image_set, args)
    raise ValueError(f'dataset {args.dataset_file} not supported')

def build_dataloader(dataset, image_set, args):
    if args.distributed:
        shuffle = True if image_set == 'train' else False
        sampler = torch.utils.data.DistributedSampler(dataset, shuffle=shuffle)
    else:
        if image_set == 'train':
            sampler = torch.utils.data.RandomSampler(dataset)
        elif image_set == 'val':
            sampler = torch.utils.data.SequentialSampler(dataset)
    
    collate_fn = CollateFN()
    if image_set == 'train':
        batch_sampler = torch.utils.data.BatchSampler(sampler, args.batch_size, drop_last=True)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_sampler=batch_sampler, collate_fn=collate_fn, num_workers=args.num_workers
        )
    elif image_set == 'val':
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, sampler=sampler, collate_fn=collate_fn, num_workers=args.num_workers
        )

    return dataloader, sampler
