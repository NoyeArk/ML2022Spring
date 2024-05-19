import torch
import pandas as pd
from torch.utils.data import DataLoader

from model import My_Model
from COVID19Dataset import COVID19Dataset
from utils import same_seed, train_valid_split, select_feat, trainer

device = 'cuda' if torch.cuda.is_available() else 'cpu'
config = {
    'seed': 5201314,     # Your seed number, you can pick your lucky number. :)
    'select_all': False,   # Whether to use all features.
    'valid_ratio': 0.2,    # validation_size = train_size * valid_ratio
    'n_epochs': 3000,     # Number of epochs.
    'batch_size': 256,
    'learning_rate': 1e-4,
    'early_stop': 400,    # If model has not improved for this many consecutive epochs, stop training.
    'save_path': './models/model.ckpt'  # Your model will be saved here.
}

if __name__ == '__main__':
    same_seed(config['seed'])  # 设置随机数种子

    train_data = pd.read_csv('dataset/covid.train.csv')
    test_data = pd.read_csv('dataset/covid.test.csv')

    train_data, valid_data = train_valid_split(train_data, config['valid_ratio'], config['seed'])

    # 打印数据集大小
    print(f'train_data size: {train_data.shape}')
    print(f'valid_data size: {valid_data.shape}')
    print(f'test_data size: {test_data.shape}')

    # 选择特征
    x_train, x_valid, x_test, y_train, y_valid = select_feat(train_data, valid_data, test_data, config['select_all'])

    # 打印特征的数量
    print(f'number of features: {x_train.shape[1]}')

    train_dataset = COVID19Dataset(x_train, y_train)
    valid_dataset = COVID19Dataset(x_valid, y_valid)
    test_dataset = COVID19Dataset(x_test)

    # Pytorch数据加载程序将Pytorch数据集加载到批中
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)

    model = My_Model(input_dim=x_train[1]).to(device)
    trainer(train_loader, valid_loader, model, config, device)
