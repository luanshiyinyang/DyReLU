import os
import random
from glob import glob

from utils.load_config import Config
random.seed(2020)


config = Config()
print(config.split)
ds_path = config.dataset_folder
txt_path = config.txt_path
if not os.path.exists(txt_path):
    os.mkdir(txt_path)


def gen_label():
    categories = os.listdir(ds_path)
    categories.sort()
    with open(os.path.join(txt_path, 'labels.txt'), 'w', encoding='utf-8') as f:
        for i in categories:
            f.write(i)
            f.write('\n')


def dataset_split():
    categories = os.listdir(ds_path)
    categories.sort()
    train_set, val_set, test_set = [], [], []
    train_labels, val_labels, test_labels = [], [], []
    index = 0
    for category in categories:
        files_path = os.path.join(ds_path, category)
        files = glob(files_path + '/*.jpg')
        random.shuffle(files)
        num = len(files)
        # train, val, test
        train_size, val_size = int(config.split[0] * num), int(config.split[1] * num)
        train_files = files[:train_size]
        val_files = files[train_size:train_size+val_size]
        test_files = files[train_size+val_size:]
        train_set.extend(train_files)
        train_labels.extend([index] * len(train_files))
        val_set.extend(val_files)
        val_labels.extend([index] * len(val_files))
        test_set.extend(test_files)
        test_labels.extend([index] * len(test_files))

        index += 1
    with open(os.path.join(txt_path, 'train.txt'), 'w', encoding='utf-8') as f:
        for i in range(len(train_set)):
            f.write(train_set[i] + " " + str(train_labels[i]))
            f.write("\n")
    with open(os.path.join(txt_path, 'val.txt'), 'w', encoding='utf-8') as f:
        for i in range(len(val_set)):
            f.write(val_set[i] + " " + str(val_labels[i]))
            f.write("\n")
    with open(os.path.join(txt_path, 'test.txt'), 'w', encoding='utf-8') as f:
        for i in range(len(test_set)):
            f.write(test_set[i] + " " + str(test_labels[i]))
            f.write("\n")


if __name__ == '__main__':
    gen_label()
    dataset_split()