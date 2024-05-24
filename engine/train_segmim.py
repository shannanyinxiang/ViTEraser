import torch 
import torch.cuda.amp as amp

from typing import Iterable 
from utils.dist import reduce_dict
from utils.logger import MetricLogger, SmoothedValue

def train_one_epoch(
      model: torch.nn.Module, 
      criterion: torch.nn.Module, 
      data_loader: Iterable, 
      optimizer,
      scaler, 
      epoch: int, 
      lr_scheduler, 
      args = None):
     
    model.train()
    criterion.train()

    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    
    num_steps = len(data_loader)

    for idx, data in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        
        images = data['image'].cuda(non_blocking=True)
        masks = data['mask'].cuda(non_blocking=True) if 'mask' in data else None
        gt_masks = data['gt_mask'].cuda(non_blocking=True)

        with amp.autocast(enabled=True):
            outputs = model(images, masks)
            loss_dict = criterion(outputs, images, masks, gt_masks)
            loss = sum(loss_dict.values())
            
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        if args.clip_max_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_max_norm)
        scaler.step(optimizer)
        scaler.update()
        lr_scheduler.step_update(epoch * num_steps + idx)

        loss_dict_reduced = reduce_dict(loss_dict)
        losses_reduced = sum(loss_dict_reduced.values())

        loss_value = losses_reduced.item()
    
        metric_logger.update(loss=loss_value, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}