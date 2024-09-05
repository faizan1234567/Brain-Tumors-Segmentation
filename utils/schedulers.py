from torch.optim.lr_scheduler import _LRScheduler

class SegResNetScheduler(_LRScheduler):
    def __init__(self, optimizer, total_epochs, alpha, last_epoch=-1):
        self.total_epochs = total_epochs
        self.alpha = alpha
        super(SegResNetScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        current_epoch = self.last_epoch + 1
        factor = (1 - current_epoch / self.total_epochs) ** 0.9
        return [self.alpha * factor for _ in self.optimizer.param_groups]

class PolyDecayScheduler(_LRScheduler):
    def __init__(self, optimizer, total_epochs, initial_lr, last_epoch=-1):
        self.total_epochs = total_epochs
        self.initial_lr = initial_lr
        super(PolyDecayScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        current_epoch = self.last_epoch + 1
        factor = (1 - current_epoch / self.total_epochs) ** 0.9
        return [self.initial_lr + (self.initial_lr * factor) for _ in self.optimizer.param_groups]