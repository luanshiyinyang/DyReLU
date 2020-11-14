import os
import yaml

import torch

current_path = os.path.dirname(__file__)


def read_config(config_file="../../config/default.yaml"):
    config_file = os.path.join(current_path, config_file)
    assert os.path.isfile(config_file), "not a config file"
    print("load config file from", config_file)
    with open(config_file, 'r', encoding="utf8") as f:
        cfg = yaml.safe_load(f.read())
    return cfg


class Config(object):
    def __init__(self):
        config = read_config()

        # dataset info
        dataset_info = config['dataset']
        self.dataset_folder = dataset_info['folder']
        self.split = list(map(int, dataset_info['split'].split(':')))
        self.split = [x / sum(self.split) for x in self.split]
        self.txt_path = dataset_info['txt_path']

        # training
        training_info = config['training']
        self.epochs = training_info['epochs']
        self.batch_size = training_info['batch_size']
        self.lr = training_info['lr']
        self.model = training_info['model']
        self.device = 'cuda:{}'.format(training_info['device']) if torch.cuda.is_available() else 'cpu'

        # exp
        exp_info = config['exp']
        self.exp_name = exp_info['name']