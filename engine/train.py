import torch 

from typing import Iterable 
from utils.dist import reduce_dict
from utils.logger import MetricLogger, SmoothedValue

def train_one_epoch(
      model: torch.nn.Module, 
      discriminator: torch.nn.Module,
      criterion: torch.nn.Module, 
      data_loader: Iterable, 
      optimizer,
      epoch: int, 
      lr_schedule: list = [0], 
      args = None):
     
    model.train()
    criterion.train()
    discriminator.train()

    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    
    optimizer['D'].param_groups[0]['lr'] = lr_schedule[epoch]
    optimizer['G'].param_groups[0]['lr'] = lr_schedule[epoch]
    optimizer['G'].param_groups[1]['lr'] = lr_schedule[epoch] * args.lr_encoder_ratio

    device = torch.device(args.device)

    for data in metric_logger.log_every(data_loader, args.print_freq, header):
        
        images = data['image'].to(device)
        labels = data['label'].to(device)
        gt_masks = data['gt_mask'].to(device)

        outputs = model(images, labels, gt_masks)

        real_prob = discriminator(labels, gt_masks)
        fake_prob_D = discriminator(outputs['outputs'][-1].contiguous().detach(), gt_masks)
        D_loss = criterion.discriminator_loss(real_prob, fake_prob_D)
        optimizer['D'].zero_grad()
        D_loss.backward()
        optimizer['D'].step()

        fake_prob_G = discriminator(outputs['outputs'][-1], gt_masks)
        outputs['real_prob'] = real_prob
        outputs['fake_prob_D'] = fake_prob_D
        outputs['fake_prob_G'] = fake_prob_G

        loss_dict = criterion(outputs, gt_masks, labels)
        loss_dict['D_loss'] = D_loss
        weight_dict = {'MSR_loss': 1, 'prc_loss': 0.01, 'style_loss': 120, 'D_fake': 0.1, 'mask_loss': 1, 'D_loss': 1} 
        for k in loss_dict.keys():
            loss_dict[k] *= weight_dict[k]

        G_loss = sum([loss_dict[k] for k in loss_dict.keys() if k != 'D_loss'])
        optimizer['G'].zero_grad()
        G_loss.backward()
        optimizer['G'].step()

        loss_dict_reduced = reduce_dict(loss_dict)
        losses_reduced = sum(loss_dict_reduced.values())
        loss_value = losses_reduced.item()
        metric_logger.update(loss=loss_value, **loss_dict_reduced)
        metric_logger.update(lr=optimizer['G'].param_groups[0]["lr"])

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}