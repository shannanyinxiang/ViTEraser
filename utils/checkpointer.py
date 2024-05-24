import torch
from utils.dist import is_main_process

class Checkpointer(object):
    def __init__(self, distributed, eval):
        self.distributed = distributed 
        self.eval = eval 

    def load(self, checkpoint_path, model, discriminator=None, optimizer=None):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        if self.distributed:
            model = model.module 
            if not discriminator is None:
                discriminator = discriminator.module 

        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'], strict=not self.eval)

        if not discriminator is None and 'discriminator' in checkpoint:
            discriminator.load_state_dict(checkpoint['discriminator'])
        
        if not optimizer is None:
            if isinstance(optimizer, dict):
                if 'G' in optimizer and 'optimizer_G' in checkpoint:
                    optimizer['G'].load_state_dict(checkpoint['optimizer_G'])
                if 'D' in optimizer and 'optimizer_D' in checkpoint:
                    optimizer['D'].load_state_dict(checkpoint['optimizer_D'])
            else:
                if 'optimizer' in checkpoint:
                    optimizer.load_state_dict(checkpoint['optimizer'])

        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1
        else:
            start_epoch = 0 
        
        return start_epoch

    def save(self, checkpoint_path, model, discriminator, optimizer, epoch, args):
        if not is_main_process():
            return 

        if self.distributed:
            model = model.module 
            if not discriminator is None:
                discriminator = discriminator.module 
        
        checkpoint = {
            'model': model.state_dict(),
            'epoch': epoch,
            'args': args
        }

        if isinstance(optimizer, dict):
            checkpoint['optimizer_G'] = optimizer['G'].state_dict()     
            if 'D' in optimizer:
                checkpoint['optimizer_D'] = optimizer['D'].state_dict()
        else:
            checkpoint['optimizer'] = optimizer.state_dict()

        if not discriminator is None:
            checkpoint['discriminator'] = discriminator.state_dict()
        torch.save(checkpoint, checkpoint_path)