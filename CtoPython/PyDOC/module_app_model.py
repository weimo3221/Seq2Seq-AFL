from flask import Flask, request
import vllm
import json
import os
import torch
import argparse


def writefragment(mes):
    with open("../docset/fragment.txt", "w") as f:
        f.write(repr(mes))
        f.write("\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", "-c", type=str, default="3", help="输入显卡的序号")
    parser.add_argument("--path", "-p", type=str, default="/data2/hugo/fuzz/train_dev_l/model/2000-readelf-lstm.pth", help="模型的地址")
    parser.add_argument("--port", "-po", type=int, default=5000, help="服务器端口的序号")
    args = parser.parse_args()
    return args

args = parse_args()

# 加载服务器板块 (Loading server plate)
app = Flask(__name__)
args_list = parse_args()
device = torch.device(f"cuda:{args_list.cuda}")
# 输出预处理板块 (Output pre-processing plate)


model = None
try:
    # 这里得是绝对路径
    model = torch.load(
        args_list.path, map_location=device)
except Exception as e:
    writefragment(e)
    print(e)


def post_process(predict):
    # output [seq_len, output_size]
    result_list = list()
    for i in range(predict.size(0) // 2):
        if 1 <= predict[i] <= 12:
            result_list.append(predict[i * 2].item())
            result_list.append(predict[i * 2 + 1].item())
    return result_list



@app.route('/')
def get_output():
    # 这个函数是主要用于输入得到的input，input是16进制的字符串，如"0011223344..."，但保证长度为2的倍数
    # (This function is mainly used to get input, input is a hexadecimal string, such as "0011223344 ...")
    # (but to ensure that the length of a multiple of 2)
    data = request.args.get('input', '')
    data = data.strip().replace('\n', '').replace('\r', '')
    if data == '':
        return []
    data = [data[i * 2: i * 2 + 2] for i in range(len(data) // 2)]
    data = [int(i, 16) for i in data]
    src = torch.LongTensor([data]).to(device)
    # n_tokens 统计不是pad的个数
    h0 = torch.zeros(model.num_layers * model.num_bidi,
                     src.shape[0], model.hidden_size).to(device)
    # [num_layers * D, batch, hidden_size]
    c0 = torch.zeros(model.num_layers * model.num_bidi,
                     src.shape[0], model.hidden_size).to(device)
    output = model(src, (h0, c0)) # [batch, seq_len, output_size]
    predict = torch.argmax(output, dim=2)  # [batch, seq_len]
    result = post_process(predict[0])
    return result


if __name__ == '__main__':
    # 如果5000端口被占用了，注意换下面的port，并保持module_client.py里面的port是一致的
    # If port 5000 is occupied, be careful to change the port below and keep the port inside module_client.py the same
    app.run(port=args.port, debug=True, use_reloader=False)
