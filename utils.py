import os
import torch
import numpy as np

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(outputs, targets):

    samples = targets.shape[0]
    outputs = outputs.sum(dim=0) / 8
    outputs = outputs.argmax(dim=-1)
    targets = targets.argmax(dim=-1)
    
    # correct_samples = (outputs == targets).sum().item()
    correct_samples = (outputs == targets).sum().item()
    return correct_samples / samples


def load_checkpoint(checkpoint_dir, title, model, optimizer):
    checkpoint_path = f'{checkpoint_dir}/{title}.pth'
    state_dict = torch.load(checkpoint_path)
    start_epoch = state_dict['epoch']
    model.load_state_dict(state_dict['model'])
    optimizer.load_state_dict(state_dict['optimizer'])
    return start_epoch


def save_checkpoint(checkpoint_dir, title, model, optimizer, epoch):
    os.makedirs(checkpoint_dir, exist_ok=True)
    state_dict = {
        'epoch': epoch + 1,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    checkpoint_path = f'{checkpoint_dir}/{title}.pth'
    torch.save(state_dict, checkpoint_path)