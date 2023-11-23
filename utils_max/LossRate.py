class LossRate():
    def __init__(self, epoch_start=0, epoch_end=120, start_lr=1, end_lr=1):
        self.epoch_start = epoch_start
        self.epoch_end = epoch_end
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.m = (self.end_lr-start_lr) / (self.epoch_end-self.epoch_start)
        self.b = self.start_lr

    def get_lossrate(self, current_epoch):
        lr = self.start_lr
        epoch = current_epoch - self.epoch_start
        if epoch > 0:
            lr = (current_epoch + self.epoch_start - self.epoch_end) * self.m + self.b
        if epoch > self.epoch_end -self.epoch_start:
            lr = self.end_lr
        return lr

