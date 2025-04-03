import torch
import torch.utils.data
from torch import nn
import pandas as pd
import timm.scheduler

pad = 499


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, samples, targets, device):
        self.samples = torch.LongTensor(samples).to(device)  # [data_size, T1]
        self.labels = torch.LongTensor(targets).to(
            device)  # [data_size, T2 + 1] 实际训练时只是取T2

    def __getitem__(self, index):
        return self.samples[index], self.labels[index]

    def __len__(self):
        return len(self.samples)


def get_dataloader(batch_size, seed_data):
    """
    获取loader的信息
    batch_size: batch_size的大小
    输出: 按批量处理好的data_loader
    """

    data_loader = torch.utils.data.DataLoader(seed_data,
                                              batch_size,
                                              shuffle=True)
    return data_loader
