# DyReLU

> 本文再DyReLU的源码基础上，将其应用到ResNet50进行对比实验。

## 数据集
Caltech256数据集中有30,607张图像，涵盖257个对象类别。对象类别非常多样，从蚱蜢🦗到音符🎵。每个类别的图像数量分布是：
- Min：80
- Med：100
- Mean：119
- Max：827

## 数据准备

在正确修改配置文件的前提下，执行下面的命令（后续命令也如此）。

```python
python dataset_preprocess.py
```

## 训练

```python
python train.py
```