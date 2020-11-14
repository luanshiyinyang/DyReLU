import os
from glob import glob

from tensorboard.backend.event_processing import event_accumulator


def get_exp_num(log_path):
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    num = len(os.listdir(log_path))
    return os.path.join(log_path, "exp{}".format(num+1))


def read_tfevents(filepath="../../runs/exp-resnet/"):
    filename = glob(filepath + '/events*')[0]

    ea = event_accumulator.EventAccumulator(filename)
    ea.Reload()
    train_loss = [x.value for x in ea.scalars.Items('train/loss')]
    val_loss = [x.value for x in ea.scalars.Items('val/loss')]
    return train_loss, val_loss


if __name__ == '__main__':
    read_tfevents()