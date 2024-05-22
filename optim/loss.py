import torch
import torch.nn as nn 
import torch.nn.functional as F

class TextRemovalLoss(nn.Module):
    def __init__(self, args):
        super(TextRemovalLoss, self).__init__()

    def forward(self, preds, mask_gt, gt):
        mask_loss = self.mask_loss(preds['mask'], 1 - mask_gt)
        D_fake = -torch.mean(preds['fake_prob_G'])
        msr_loss = self.MSR_loss(preds['outputs'], mask_gt, gt)
        prc_loss = self.percetual_loss(preds['feat_output_comp'], preds['feat_output'], preds['feat_gt'])
        style_loss = self.style_loss(preds['feat_output_comp'], preds['feat_output'], preds['feat_gt'])

        losses = {'MSR_loss': msr_loss, 'prc_loss': prc_loss, 'style_loss': style_loss,
                  'D_fake': D_fake, 'mask_loss': mask_loss}
        return losses

    def mask_loss(self, mask_pred, mask_label):
        return dice_loss(mask_pred, mask_label)
    
    @staticmethod
    def discriminator_loss(real_prob, fake_prob):
        return hinge_loss(real_prob, 1) + hinge_loss(fake_prob, -1)
    
    def percetual_loss(self, feat_output_comp, feat_output, feat_gt):
        pcr_losses = []
        for i in range(3):
            pcr_losses.append(F.l1_loss(feat_output[i], feat_gt[i]))
            pcr_losses.append(F.l1_loss(feat_output_comp[i], feat_gt[i]))
        return sum(pcr_losses)
    
    def style_loss(self, feat_output_comp, feat_output, feat_gt):
        style_losses = []
        for i in range(3):
            style_losses.append(F.l1_loss(gram_matrix(feat_output[i]), gram_matrix(feat_gt[i])))
            style_losses.append(F.l1_loss(gram_matrix(feat_output_comp[i]), gram_matrix(feat_gt[i])))
        return sum(style_losses)

    def MSR_loss(self, outputs, mask, gt, scale_factors=[0.25, 0.5, 1.0], weights=[[5, 0.8], [6, 1], [10, 2]]):
        msr_losses = []
        for output, scale_factor, weight in zip(outputs, scale_factors, weights):
            if scale_factor != 1:
                mask_ = F.interpolate(mask, scale_factor=scale_factor, recompute_scale_factor=True)
                gt_ = F.interpolate(gt, scale_factor=scale_factor, recompute_scale_factor=True)
            else:
                mask_ = mask; gt_ = gt 
            msr_losses.append(weight[0] * F.l1_loss((1 - mask_) * output, (1 - mask_) * gt_))
            msr_losses.append(weight[1] * F.l1_loss(mask_ * output, mask_ * gt_))
        return sum(msr_losses)
            
def gram_matrix(feat):
    (b, ch, h, w) = feat.size()
    feat = feat.view(b, ch, h * w)
    feat_t = feat.transpose(1, 2)
    gram = torch.bmm(feat, feat_t) / (ch * h * w)
    return gram

def hinge_loss(input, target):
    return torch.mean(F.relu(1 - target * input))

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