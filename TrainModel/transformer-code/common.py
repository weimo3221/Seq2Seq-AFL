import torch
import torch.utils.data
from torch import nn
import pandas as pd
import timm.scheduler
import math

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


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 初始化Shape为(max_len, d_model)的PE (positional encoding)
        pe = torch.zeros(max_len, d_model)
        # 初始化一个tensor [[0, 1, 2, 3, ...]]
        position = torch.arange(0, max_len).unsqueeze(1)
        # 这里就是sin和cos括号中的内容，通过e和ln进行了变换
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        # 计算PE(pos, 2i)
        pe[:, 0::2] = torch.sin(position * div_term)
        # 计算PE(pos, 2i+1)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 为了方便计算，在最外面在unsqueeze出一个batch
        pe = pe.unsqueeze(0)
        # 如果一个参数不参与梯度下降，但又希望保存model的时候将其保存下来
        # 这个时候就可以用register_buffer
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x 为embedding后的inputs，例如(1,7, 128)，batch size为1,7个单词，单词维度为128
        """
        # 将x和positional encoding相加。
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)


class TransformerModel(nn.Module):

    def __init__(self, d_model=128, vocab_size=500, num_layers=2, n_head=8, pad_id=pad):
        super(TransformerModel, self).__init__()

        # 定义词向量，词典数为10。我们不预测两位小数。
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=d_model, padding_idx=pad_id)
        # 定义Transformer
        # self.transformer = nn.Transformer(d_model=d_model, num_encoder_layers=num_layers,
        #                                   num_decoder_layers=num_layers,
        #                                   dim_feedforward=512,
        #                                   batch_first=True)

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head, batch_first=True)
        self.encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=num_layers)

        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_head, batch_first=True)
        self.decoder = nn.TransformerDecoder(
            self.decoder_layer, num_layers=num_layers)
        # 定义位置编码器
        self.positional_encoding = PositionalEncoding(d_model, dropout=0)

        # 定义最后的线性层，这里并没有用Softmax，因为没必要。
        # 因为后面的CrossEntropyLoss中自带了
        self.predictor = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt):
        # 生成mask
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            tgt.size()[-1]).to(src.device)  # [T2, T2]
        src_key_padding_mask = TransformerModel.get_key_padding_mask(
            src).type(torch.bool).to(src.device)  # [B, T1]
        tgt_key_padding_mask = TransformerModel.get_key_padding_mask(
            tgt).type(torch.bool).to(src.device)  # [B, T2]

        # 对src和tgt进行编码
        src = self.embedding(src)  # [B, T1, E]
        tgt = self.embedding(tgt)  # [B, T2, E]
        # 给src和tgt的token增加位置信息
        src = self.positional_encoding(src)  # [B, T1, E]
        tgt = self.positional_encoding(tgt)  # [B, T2, E]

        # 将准备好的数据送给transformer
        memory = self.encoder(
            src, src_key_padding_mask=src_key_padding_mask)  # [B, T1, E]
        out = self.decoder(tgt, memory, tgt_mask=tgt_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask)  # [B, T2, E]
        # out = self.transformer(src, tgt,
        #                        tgt_mask=tgt_mask,
        #                        src_key_padding_mask=src_key_padding_mask,
        #                        tgt_key_padding_mask=tgt_key_padding_mask)

        """
        这里直接返回transformer的结果。因为训练和推理时的行为不一样，
        所以在该模型外再进行线性层的预测。
        """
        return out

    @staticmethod
    def get_key_padding_mask(tokens):
        """
        用于key_padding_mask
        """
        key_padding_mask = torch.zeros(tokens.size())
        key_padding_mask[tokens == pad] = 1
        return key_padding_mask
