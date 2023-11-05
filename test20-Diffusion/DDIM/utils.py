import math
import matplotlib.pyplot as plt
import numpy as np


class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]

        print('\r' + '\t'.join(entries), end='')

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def make_grid(sample, sampling_number=64):
    row_number = int(math.sqrt(sampling_number))
    col_number = int(math.sqrt(sampling_number))
    sample = sample[:sampling_number].detach().cpu().numpy()
    shape = sample.shape
    show_sample = np.zeros([row_number * shape[2], col_number * shape[3]]).astype(np.float32)
    for row in range(row_number):
        for col in range(col_number):
            sample_ = sample[row + col * row_number][0]
            show_sample[row * shape[2]: (row + 1) * shape[2], col * shape[3]: (col + 1) * shape[3]] = (
                                                                                                                  sample_ - sample_.min()) / (
                                                                                                                  sample_.max() - sample_.min()) * 255

    show_sample = show_sample.astype(np.uint8)
    return show_sample


def imshow(sample, sampling_number=64, save=False, name=None):
    plt.figure(figsize=(10, 10))
    show_sample = make_grid(sample, sampling_number)

    plt.axis(False)
    plt.imshow(show_sample, cmap='gray')
    if save:
        if name is None:
            name = 'sample.png'
        plt.savefig(name)
    plt.show()
