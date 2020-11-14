import os

import matplotlib.pyplot as plt
import numpy as np

from utils.utils import read_tfevents


def plot_history():
    base_dir = '../runs/'
    plot_exp = ['exp-resnet', 'exp-resnet-drelu']
    plt.figure(figsize=(12, 6))
    for exp in plot_exp:
        train_loss, val_loss = read_tfevents(os.path.join(base_dir, exp))
        # plt.plot(np.arange(len(train_loss)), train_loss, label=exp)
        plt.plot(np.arange(len(val_loss)), val_loss, label=exp)
    plt.legend(loc='best')
    plt.title('validation loss')
    plt.savefig("../assets/validation_loss.png")
    plt.show()


if __name__ == '__main__':
    plot_history()
