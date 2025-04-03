import torch
import torch.nn as nn
import numpy as np
from common import MyDataset, get_dataloader, pad
import pandas as pd
import argparse
from datetime import datetime
import os
import matplotlib.pyplot as plt


# data_size: 数据集的大小
# T是src和target的长度seq_len，这两个的长度应该是一致的
# B: batch_size
# H: hidden_dim
# E: embedding_dim
# V: vocab_size
# O: output_size
# D: num_bidi


# 从csv文件中读取内容并构建数据集的函数
def dataset_construct(df, device):
    src = df["src"]
    target = df["target"]
    for i in range(len(src)):
        src[i] = src[i].split(',')
        src[i] = [int(x) for x in src[i]]
        target[i] = target[i].split(',')
        target[i] = [int(x) for x in target[i]]
    src = list(src)
    target = list(target)

    data_x = np.array(src, dtype=np.float32)
    data_y = np.array(target, dtype=np.float32)
    print(data_x.shape)
    print(data_y.shape)
    # data_x shape: (data_size, seq_len)
    # data_y shape: (data_size, actual_len)
    dataset = MyDataset(data_x, data_y, device)
    return dataset


# 定义LSTM深度学习模型
class MyLSTM(nn.Module):
    """
        Parameters：
        - input_size: feature size
        - hidden_size: number of hidden units
        - output_size: number of output
        - num_layers: layers of LSTM to stack
    """

    def __init__(self, hidden_size=1, output_size=1, embed_size=1, num_layers=1, bidirectional=False,
                 device=torch.device("cpu")):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.embed_size = embed_size
        self.padding_idx = pad
        self.vocab_size = pad + 1
        self.bidirectional = bidirectional
        self.embed = nn.Embedding(
            self.vocab_size, embed_size, padding_idx=self.padding_idx).to(device)
        if bidirectional is False:
            # 利用torch.nn进行LSTM的加载
            self.lstm = nn.LSTM(embed_size, hidden_size,
                                num_layers, batch_first=True).to(device)
            self.num_bidi = 1
        else:
            if num_layers == 1:
                self.lstm = nn.LSTM(
                    embed_size, hidden_size, num_layers, batch_first=True, bidirectional=True).to(device)
            else:
                self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True, bidirectional=True,
                                    dropout=0.3).to(device)
            self.num_bidi = 2
            # 利用torch.nn进行双向LSTM的加载
        self.linear1 = nn.Linear(
            hidden_size * self.num_bidi, output_size).to(device)  # 全连接层 貌似这里

    def forward(self, _x, state):
        # _x: [batch, seq_len] -> [batch, seq_len, embed_size]
        _x = self.embed(_x)
        # state: ([num_layers, batch, hidden_size], [num_layers, batch, hidden_size])
        x, (h_n, c_n) = self.lstm(_x, state)
        # x: [batch, seq_len, hidden_size * D]
        # h_n: [num_layers * D, batch, hidden_size], c_n: [num_layers * D, batch, hidden_size]
        b, s, h = x.shape
        # b: batch, s: seq_len, h: hidden_size
        x = self.linear1(x)
        # x: [batch, seq_len, hidden_size] -> [batch, seq_len, output_size]
        # x[-1, :, :]
        return x


# 设置初始化命令参数的函数
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dataset", "-train", type=str, default=r"../dataset/train_dataset_lstm.csv",
                        help="csv文件路径")
    parser.add_argument("--test_dataset", "-test", type=str, default=r"../dataset/test_dataset_lstm.csv",
                        help="csv文件路径")
    parser.add_argument("--program", "-p", type=str,
                        default="nm", help="选择程序的类型")
    parser.add_argument("--cuda", "-c", type=int, default=0, help="选择cuda的类型")
    parser.add_argument("--type", "-t", type=int, default=0,
                        help="选择训练的类型，0为单向LSTM，1为双向LSTM")
    args = parser.parse_args()
    return args


# 训练LSTM模型的函数
def train_lstm(model, train_loader, test_loader, lr, epochs, device):
    train_loss_all = []
    train_acc_all = []
    val_loss_all = []
    val_acc_all = []
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criteria = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        print('-' * 10)
        print("Epoch {}/{}".format(epoch + 1, epochs))
        # 每个epoch有两个阶段,训练阶段和验证阶段
        train_loss = 0.0
        train_corrects = 0
        val_loss = 0.0
        val_corrects = 0

        # 训练阶段
        model.train()
        num = 0
        for step, batch in enumerate(train_loader):
            src = batch[0]  # [batch, seq_len]
            tgt = batch[1]  # [batch, seq_len]
            # 创建一个掩码，其中 pad 位置为 0，其他位置为 1
            mask = src.ne(pad).float()  # [batch, seq_len]
            # n_tokens 统计不是pad的个数
            h0 = torch.zeros(model.num_layers * model.num_bidi,
                             src.shape[0], model.hidden_size).to(device)
            # [num_layers * D, batch, hidden_size]
            c0 = torch.zeros(model.num_layers * model.num_bidi,
                             src.shape[0], model.hidden_size).to(device)
            # [num_layers * D, batch, hidden_size]
            # 清空梯度
            optimizer.zero_grad()
            # 进行lstm的计算
            output = model(src, (h0, c0))  # [batch, seq_len, output_size]
            """
                计算损失。由于训练时我们的是对所有的输出都进行预测，所以需要对out进行reshape一下。
                    我们的out的Shape为(batch, seq_len, output_size)，view之后变为：
                    (batch_size*seq_len, output_size)。
                    而在这些预测结果中，我们只需要对非<pad>部分进行，所以需要进行正则化。也就是
                    除以n_tokens。
            """
            loss = torch.tensor(0.0).to(device)
            for i in range(src.size(0)):
                r_num = mask[i, :].sum()
                loss += criteria(output[i, :int(r_num), :].contiguous().view(-1, output[i, :int(r_num), :].size(-1)),
                                 tgt[i, :int(r_num)].contiguous().view(-1))
            loss /= mask.sum()
            # 计算梯度
            loss.backward()
            # 更新参数
            optimizer.step()

            # 计算模型输出的值predict
            predict = torch.argmax(output, dim=2)  # [batch, seq_len]
            equal_elements = torch.eq(predict, tgt)
            num_equal = (equal_elements * mask).sum()
            train_corrects += (num_equal / mask.sum()).cpu()
            train_loss += loss.item()
            num += 1
        train_loss_all.append(train_loss / num)
        train_acc_all.append(train_corrects / num)
        # train_loss_all: [epoch]
        # train_acc_all: [epoch]
        print('{} Train Loss: {:.8f}  Train Acc: {:.8f}'.format(
            epoch, train_loss_all[-1], train_acc_all[-1]))

        model.eval()  # 设置模型为训练模式评估模式
        num = 0
        for step, batch in enumerate(test_loader):
            src = batch[0]  # [batch, seq_len]
            tgt = batch[1]  # [batch, seq_len]
            # 创建一个掩码，其中 pad 位置为 0，其他位置为 1
            mask = src.ne(pad).float()
            # n_tokens 统计不是pad的个数
            h0 = torch.zeros(model.num_layers * model.num_bidi,
                             src.shape[0], model.hidden_size).to(device)
            # [num_layers * D, batch, hidden_size]
            c0 = torch.zeros(model.num_layers * model.num_bidi,
                             src.shape[0], model.hidden_size).to(device)
            # [num_layers * D, batch, hidden_size]
            # 进行lstm的计算
            output = model(src, (h0, c0))  # [batch, seq_len, output_size]
            """
                计算损失。由于训练时我们的是对所有的输出都进行预测，所以需要对out进行reshape一下。
                    我们的out的Shape为(batch, seq_len, output_size)，view之后变为：
                    (batch_size*seq_len, output_size)。
                    而在这些预测结果中，我们只需要对非<pad>部分进行，所以需要进行正则化。也就是
                    除以n_tokens。
            """
            loss = torch.tensor(0.0).to(device)
            for i in range(src.size(0)):
                r_num = mask[i, :].sum()
                loss += criteria(output[i, :int(r_num), :].contiguous().view(-1, output[i, :int(r_num), :].size(-1)),
                                 tgt[i, :int(r_num)].contiguous().view(-1))
            loss /= mask.sum()
            # 计算模型输出的值predict
            predict = torch.argmax(output, dim=2)  # [batch, seq_len]
            equal_elements = torch.eq(predict, tgt)
            num_equal = (equal_elements * mask).sum()
            val_corrects += (num_equal / mask.sum()).cpu()
            val_loss += loss.item()
            num += 1
        # 计算一个epoch在训练集上的损失和精度
        val_loss_all.append(val_loss / num)
        val_acc_all.append(val_corrects / num)
        # val_loss_all: [epoch]
        # val_acc_all: [epoch]
        print('{} Val Loss: {:.8f}  Val Acc: {:.8f}'.format(
            epoch, val_loss_all[-1], val_acc_all[-1]))
    train_process = pd.DataFrame(
        data={"epoch": range(epochs),
              "train_loss_all": train_loss_all,
              "train_acc_all": train_acc_all,
              "val_loss_all": val_loss_all,
              "val_acc_all": val_acc_all})
    return model, train_process


def main():
    # mylstm = MyLSTM(hidden_size=10, output_size=4, embed_size=10, num_layers=2)
    # input = torch.tensor([[2, 2, 3, 3], [1, 2, 5, 4], [3, 7, 6, 8]], dtype=torch.long)
    # h0 = torch.zeros(2, 3, 10)
    # c0 = torch.zeros(2, 3, 10)
    # output = mylstm(input, (h0, c0))
    # torch.optim.AdamW(mylstm.parameters(), lr=0.001)
    # optimizer = loss_function = nn.MSELoss()
    # epochs = 150

    args = parse_args()
    batch_size = 4
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.cuda}")
        print(f"训练的设备为cuda:{args.cuda}")
    else:
        device = torch.device("cpu")
        print(f"训练的设备为cpu")
    df_train = pd.read_csv(args.train_dataset)
    df_test = pd.read_csv(args.test_dataset)
    train_dataset = dataset_construct(df_train, device)
    test_dataset = dataset_construct(df_test, device)
    train_loader = get_dataloader(batch_size, train_dataset)
    test_loader = get_dataloader(batch_size, test_dataset)

    start_time = datetime.now()
    lr = 0.005
    if args.type == 0:
        model = MyLSTM(hidden_size=128, output_size=2,
                       embed_size=128, num_layers=2, bidirectional=False, device=device)
        model_result, train_process = train_lstm(
            model, train_loader, test_loader, lr=lr, epochs=200, device=device)
    elif args.type == 1:
        model = MyLSTM(hidden_size=128, output_size=2,
                       embed_size=128, num_layers=2, bidirectional=True, device=device)
        model_result, train_process = train_lstm(
            model, train_loader, test_loader, lr=lr, epochs=200, device=device)
    else:
        model_result = None
        train_process = None
        exit()
    end_time = datetime.now()

    if not os.path.exists("../model"):
        os.mkdir("../model")
    if not os.path.exists("../progress"):
        os.mkdir("../progress")

    torch.save(
        model_result, f'../model/{end_time.strftime("%y%m%d-%H-%M-%S")}-{args.program}-lstm.pth')
    csv_name = f'../progress/{args.program}-lstm-process-{end_time.strftime("%y%m%d-%H-%M-%S")}.csv'
    print("Save in:",
          f'../model/{end_time.strftime("%y%m%d-%H-%M-%S")}-{args.program}-lstm.pth')

    progress_name = f'../progress/{args.program}-lstm-process-{end_time.strftime("%y%m%d-%H-%M-%S")}.txt'
    with open(progress_name, "w") as f:
        f.write(f'训练模型: LSTM\n')
        f.write(f'训练的数据集: {args.program}\n')
        f.write(f'训练模型类型: {args.type}\n')
        f.write(f'训练模型的学习率: {lr}\n')
        f.write(f'训练时间: {str(end_time - start_time)}\n')

    # 保存训练过程
    train_process.to_csv(csv_name, index=False)
    # 可视化模型训练过程中
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 2, 1)
    plt.plot(train_process.epoch, train_process.train_loss_all,
             "r.-", label="Train loss")
    plt.plot(train_process.epoch, train_process.val_loss_all,
             "bs-", label="Val loss")
    plt.legend()
    plt.xlabel("Epoch number", size=13)
    plt.ylabel("Loss value", size=13)
    plt.subplot(1, 2, 2)
    plt.plot(train_process.epoch, train_process.train_acc_all,
             "r.-", label="Train acc")
    plt.plot(train_process.epoch, train_process.val_acc_all,
             "bs-", label="Val acc")
    plt.xlabel("Epoch number", size=13)
    plt.ylabel("Acc", size=13)
    plt.legend()
    plt.show()


def test():
    a = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    model = MyLSTM(hidden_size=128, output_size=2,
                   embed_size=128, num_layers=2, bidirectional=True)
    h0 = torch.zeros(model.num_layers * 2, 3, model.hidden_size)
    c0 = torch.zeros(model.num_layers * 2, 3, model.hidden_size)
    output = model(a, (h0, c0))
    print(output.size())


if __name__ == '__main__':
    # test()
    main()
