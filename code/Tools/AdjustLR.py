import math

def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 2 epochs"""
    # lr *= 1 / (epoch**0.5)
    DELACY = 0.005
    epoch = epoch * DELACY + 1.0
    lr *= 2 / (1.0 + math.e**(epoch)) + (1 - 2 /
                                         (1 + math.e**(epoch))) / (epoch**0.5)
    if not math.isnan(lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr