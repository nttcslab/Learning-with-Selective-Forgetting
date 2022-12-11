# -*- coding: utf-8 -*-

from torch.optim.lr_scheduler import _LRScheduler, StepLR, MultiStepLR, LambdaLR

class WarmupConstantSchedule(LambdaLR):
    def __init__(self, optimizer, warmup_steps, last_epoch=-1):

        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1.0, warmup_steps))
            return 1.

        super(WarmupConstantSchedule, self).__init__(optimizer, lr_lambda, last_epoch=last_epoch)

class lr_scheduling_func(LambdaLR):
    def __init__(self, optimizer, milestones, gamma):

        def lr_lambda(epoch):
            
            n = int(epoch) % int(num_epoch)
            return (gamma**n)

        super(lr_scheduling_func, self).__init__(optimizer, lr_lambda)

class lr_scheduling_func_step(LambdaLR):
    def __init__(self, optimizer, milestones, gamma):

        def lr_lambda(epoch):            
            n = sum( [1 for k in milestones if k < epoch] )
            return (gamma**n)

        super(lr_scheduling_func_step, self).__init__(optimizer, lr_lambda)

class FindLR(_LRScheduler):
    """exponentially increasing learning rate
    Args:
        optimizer: optimzier(e.g. SGD)
        num_iter: totoal_iters 
        max_lr: maximum  learning rate
    """
    def __init__(self, optimizer, max_lr=10, num_iter=100, last_epoch=-1):
        
        self.total_iters = num_iter
        self.max_lr = max_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):

        return [base_lr * (self.max_lr / base_lr) ** (self.last_epoch / (self.total_iters + 1e-32)) for base_lr in self.base_lrs]


class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):
        
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]        