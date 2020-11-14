import os

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from runx.logx import logx

from utils.load_config import Config
from dataset import MyDataset, get_tfms
from models.resnet import ResNet50
from models.resnet_dy import ResNet50_dy
from utils.utils import get_exp_num

config = Config()
logdir = get_exp_num("../runs/") if config.exp_name == 'None' else "../runs/{}".format(config.exp_name)
logx.initialize(logdir, coolname=True, tensorboard=True)


def train_epoch(epoch):
    model.train()
    losses = 0.0
    for step, (x, y) in enumerate(train_loader):
        x, y = x.to(config.device), y.to(config.device)
        pred = model(x)
        loss = criterion(pred, y)
        losses += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            logx.msg("epoch {} step {} training loss {}".format(epoch, step, loss.item()))
    logx.msg("epoch {} training loss {}".format(epoch, losses))
    logx.metric("train", {"loss": losses / (step + 1)})
    return losses


def test_epoch(epoch):
    model.eval()
    losses = 0.0
    with torch.no_grad():
        for step, (x, y) in enumerate(val_loader):
            x, y = x.to(config.device), y.to(config.device)
            pred = model(x)
            loss = criterion(pred, y)
            losses += loss
    save_dict = {
        'state_dict': model.state_dict()
    }
    logx.msg("epoch {} validation loss {}".format(epoch, losses))
    logx.metric('val', {'loss': losses / (step + 1)})
    logx.save_model(save_dict, losses, epoch, higher_better=False, delete_old=True)


# dataset


tfms = get_tfms(224)
train_ds = MyDataset(txt=os.path.join(config.txt_path, 'train.txt'), transform=tfms)
val_ds = MyDataset(txt=os.path.join(config.txt_path, 'train.txt'), transform=tfms)
train_loader = DataLoader(dataset=train_ds, batch_size=config.batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_ds, batch_size=config.batch_size, shuffle=False)
print("data load successfully")
# model
if config.model == "resnet":
    model = ResNet50()
else:
    model = ResNet50_dy()
model = model.to(config.device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, nesterov=True)

for i in range(config.epochs):
    train_epoch(i)
    test_epoch(i)