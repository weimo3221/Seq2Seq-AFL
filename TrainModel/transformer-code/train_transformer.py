import torch
from common import MyDataset, get_dataloader, pad, TransformerModel
import torch.utils.data
from torch import nn
import pandas as pd
import timm.scheduler
import argparse
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime


# data_size: 数据集的大小
# T1是src的长度seq_len，T2是target的长度seq_len
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dataset", "-train", type=str, default=r"../queue_data/csv文件/train_dataset_lstm.csv",
                        help="csv文件路径")
    parser.add_argument("--test_dataset", "-test", type=str, default=r"../queue_data/csv文件/test_dataset_lstm.csv",
                        help="csv文件路径")
    parser.add_argument("--program", "-p", type=str,
                        default="nm", help="选择程序的类型")
    parser.add_argument("--cuda", "-c", type=int, default=0, help="选择cuda的类型")
    args = parser.parse_args()
    return args


def train_transformer(model, trainloader, testloader, lr, num_epochs, device):
    train_loss_all = []
    train_acc_all = []
    val_loss_all = []
    val_acc_all = []
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = timm.scheduler.CosineLRScheduler(optimizer=optimizer,
                                                 t_initial=num_epochs,
                                                 lr_min=1e-5,
                                                 warmup_t=75,
                                                 warmup_lr_init=1e-4
                                                 )
    # warmup的值是算总epoch的10%-20%
    criteria = nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        print('-' * 10)
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        # 调整优化器的学习率
        scheduler.step(epoch)
        # 每个epoch有两个阶段,训练阶段和验证阶段
        train_loss = 0.0
        train_corrects = 0
        val_loss = 0.0
        val_corrects = 0
        # 训练阶段
        model.train()
        num = 0
        for step, batch in enumerate(trainloader):
            src = batch[0]  # [B, T1]
            tgt = batch[1]  # [B, T2 + 1] 这里输出值最开始有个0，所以会加1
            tgt_y = tgt[:, 1:]  # [B, T2]
            tgt = tgt[:, :-1]  # [B, T2]

            # 创建一个掩码，其中 pad 位置为 0，其他位置为 1
            mask = tgt_y.ne(pad).float()  # [B, T2]
            # 清空梯度
            optimizer.zero_grad()
            # 进行transformer的计算
            out = model(src, tgt)  # [B, T2, H]
            # 将结果送给最后的线性层进行预测
            out = model.predictor(out)  # [B, T2, V]
            """
            计算损失。由于训练时我们的是对所有的输出都进行预测，所以需要对out进行reshape一下。
                    我们的out的Shape为(batch_size, 词数, 词典大小)，view之后变为：
                    (batch_size*词数, 词典大小)。
                    而在这些预测结果中，我们只需要对非<pad>部分进行，所以需要进行正则化。也就是
                    除以n_tokens。
            """

            loss = torch.tensor(0.0).to(device)

            for i in range(src.size(0)):
                r_num = mask[i, :].sum()
                loss += criteria(out[i, :int(r_num), :].contiguous().view(-1, out[i, :int(r_num), :].size(-1)),
                                 tgt_y[i, :int(r_num)].contiguous().view(-1))
            loss /= mask.sum()
            # 计算梯度
            loss.backward()
            # 更新参数
            optimizer.step()
            y = torch.argmax(out, dim=2)  # [B, T2]
            equal_elements = torch.eq(y, tgt_y)
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

        # 计算一个epoch的训练后在验证集上的损失和精度
        model.eval()  # 设置模型为训练模式评估模式
        num = 0
        for step, batch in enumerate(testloader):
            src = batch[0]  # [B, T1]
            tgt = batch[1]  # [B, T2 + 1]
            tgt_y = tgt[:, 1:]  # [B, T2]
            # tgt从<bos>开始，看看能不能重新输出src中的值
            tgt = torch.LongTensor([[0]] * src.size(0)).to(device)  # [B, 1]
            # 创建一个掩码，其中 pad 位置为 0，其他位置为 1
            mask = tgt_y.ne(pad).float()  # [B, T2]
            # 一个一个词预测，直到预测为<eos>，或者达到句子最大长度
            out = None
            for j in range(tgt_y.size(1)):
                # 进行transformer计算
                out = model(src, tgt)  # [B, 1+j, H]
                # 预测结果，因为只需要看最后一个词，所以取`out[:, -1]`
                predict = model.predictor(out[:, -1])  # [B, V]
                # 找出最大值的index
                y = torch.argmax(predict, dim=1)  # [B]
                # 和之前的预测结果拼接到一起
                tgt = torch.concat([tgt, y.unsqueeze(1)], dim=1)  # [B, 1+j]
            out = model.predictor(out)  # [B, T2, V]

            loss = torch.tensor(0.0).to(device)

            for i in range(src.size(0)):
                r_num = mask[i, :].sum()
                loss += criteria(out[i, :int(r_num), :].contiguous().view(-1, out[i, :int(r_num), :].size(-1)),
                                 tgt_y[i, :int(r_num)].contiguous().view(-1))
            loss /= mask.sum()

            y = tgt[:, 1:]  # [B, T2]
            equal_elements = torch.eq(y, tgt_y)
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
        data={"epoch": range(num_epochs),
              "train_loss_all": train_loss_all,
              "train_acc_all": train_acc_all,
              "val_loss_all": val_loss_all,
              "val_acc_all": val_acc_all})
    return model, train_process


def main():
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
    lr = 0.001
    model = TransformerModel(d_model=128, vocab_size=500).to(device)
    model_result, train_process = train_transformer(model, train_loader, test_loader, lr=lr, num_epochs=300,
                                                    device=device)
    end_time = datetime.now()

    if not os.path.exists("../model"):
        os.mkdir("../model")
    if not os.path.exists("../progress"):
        os.mkdir("../progress")

    torch.save(
        model_result, f'../model/{end_time.strftime("%y%m%d-%H-%M-%S")}-{args.program}-transformer.pth')
    csv_name = f'../progress/{args.program}-transformer-process-{end_time.strftime("%y%m%d-%H-%M-%S")}.csv'
    print("Save in:",
          f'../model/{end_time.strftime("%y%m%d-%H-%M-%S")}-{args.program}-transformer.pth')

    progress_name = f'../progress/{args.program}-transformer-process-{end_time.strftime("%y%m%d-%H-%M-%S")}.txt'
    with open(progress_name, "w") as f:
        f.write(f'训练模型: Transformer\n')
        f.write(f'训练的数据集: {args.program}\n')
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


if __name__ == "__main__":
    main()
