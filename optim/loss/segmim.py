import torch
import torch.nn as nn
import torch.nn.functional as F

class SegMIMLoss(nn.Module):
    def __init__(self, args):
        super(SegMIMLoss, self).__init__()
        self.patch_size = 4

    def forward(self, outputs, images, masks, seg_masks):
        loss = {}
        if 'mim' in outputs:
            mim_preds = outputs['mim'][0]
            mim_loss = F.l1_loss(images, mim_preds, reduction='none')
            masks = masks.repeat_interleave(self.patch_size, 1).repeat_interleave(self.patch_size, 2).unsqueeze(1).contiguous()
            mim_loss = (mim_loss * masks).sum() / (masks.sum() + 1e-5) / 3
            loss['mim'] = mim_loss

        seg_preds = outputs['textseg']
        seg_loss = dice_loss(seg_preds, 1 - seg_masks)
        loss['textseg'] = seg_loss

        return loss


def dice_loss(input, target):
    input = torch.sigmoid(input)
    input = input.flatten(1)
    target = target.flatten(1)

    a = torch.sum(input * target, 1)
    b = torch.sum(input * input, 1) + 0.001
    c = torch.sum(target * target, 1) + 0.001
    dice_loss = (2 * a) / (b + c)
    dice_loss = torch.mean(dice_loss)
    return 1 - dice_loss