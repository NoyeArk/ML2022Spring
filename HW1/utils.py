import csv
import math
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import random_split

# 进度条
from tqdm import tqdm

# 绘制训练曲线
from torch.utils.tensorboard import SummaryWriter

from model import My_Model

def same_seed(seed):
    # Fixes random number generator seeds for reproducibility.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_valid_split(data_set, valid_ratio, seed):
    # 将所给的训练集划分为训练集和验证集
    valid_set_size = int(valid_ratio * len(data_set))
    train_set_size = len(data_set) - valid_set_size
    train_set, valid_set = random_split(data_set, [train_set_size, valid_set_size], generator=torch.Generator().manual_seed(seed))
    return np.array(train_set), np.array(valid_set)


def predict(test_loader, model, device):
    model.eval()  # Set your models to evaluation mode.
    preds = []
    for x in tqdm(test_loader):
        x = x.to(device)
        with torch.no_grad():
            pred = model(x)
            preds.append(pred.detach().cpu())
    preds = torch.cat(preds, dim=0).numpy()
    return preds


def select_feat(train_data, valid_data, test_data, select_all=True):
    # Selects useful features to perform regression
    y_train, y_valid = train_data[:, -1], valid_data[:, -1]
    raw_x_train, raw_x_valid, raw_x_test = train_data[:, :-1], valid_data[:, :-1], test_data

    if select_all:
        feat_idx = list(range(raw_x_train.shape[1]))
    else:
        feat_idx = [53, 69, 85, 101]  # TODO: Select suitable feature columns.

    return raw_x_train[:, feat_idx], raw_x_valid[:, feat_idx], raw_x_test[:, feat_idx], y_train, y_valid


# 定义训练过程
def trainer(train_loader, valid_loader, model, config, device):
    # 损失函数定义
    criterion = nn.MSELoss()
    # 定义优化算法
    # optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'])
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    writer = SummaryWriter(log_dir=config['tensorboard_dir'])

    n_epochs, best_loss, step, early_stop_cnt = config['n_epochs'], math.inf, 0, 0

    for epoch in range(n_epochs):
        model.train()  # 训练模式
        loss_record = []

        train_pbar = tqdm(train_loader, position=0, leave=True)

        for x, y in train_pbar:
            optimizer.zero_grad()  # 设置梯度为0
            x, y = x.to(device), y.to(device)

            pred = model(x)  # 模型预测
            loss = criterion(pred, y)  # 计算损失
            loss.backward()   # 计算梯度
            optimizer.step()  # 根据梯度更新模型参数

            step += 1
            loss_record.append(loss.detach().item())  # 记录损失

            # 显示当前迭代轮次和训练损失
            train_pbar.set_description(f'Epoch [{epoch + 1} / {n_epochs}]')
            train_pbar.set_postfix({'Loss': loss.detach().item()})

        mean_train_loss = sum(loss_record) / len(loss_record)
        writer.add_scalar('Loss/train', mean_train_loss, step)

        model.eval()  # 预测模式
        loss_record = []
        for x, y in valid_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                pred = model(x)
                loss = criterion(pred, y)
            loss_record.append(loss)

        mean_valid_loss = sum(loss_record) / len(loss_record)
        print(f'Epoch [{epoch + 1} / {n_epochs}]: Train loss: {mean_train_loss:.4f}, Valid loss: {mean_valid_loss:.4f}')
        writer.add_scalar('Loss/valid', mean_valid_loss, step)

        if mean_valid_loss < best_loss:
            best_loss = mean_valid_loss
            torch.save(model.state_dict(), config['save_path'])  # 保存模型
            print('保存模型，其损失为{:.3f}...'.format(best_loss))
            early_stop_cnt = 0
        else:
            early_stop_cnt += 1

        if early_stop_cnt >= config['early_stop']:
            print('\nModel is not improving, so we halt the training session')
            return


def save_pred(preds, file):
    # 保存预测结果
    with open(file, 'w', newline='') as fp:
        writer = csv.writer(fp)
        writer.writerow(['id', 'tested_positive'])
        for i, p in enumerate(preds):
            print(i, p)
            writer.writerow([i, p])
