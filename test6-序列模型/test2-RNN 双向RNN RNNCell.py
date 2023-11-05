# RNN layer RNNCell

import torch
import torch.nn as nn

# RNN layer(一步到位)
rnn = nn.RNN(input_size=5, hidden_size=10, num_layers=1, batch_first=True)# 有两个返回值，一个是输出，一个是最后一个隐藏层的输出，输出可以再接一个全连接层等
# input_size是输入的一个词向量的长度，hidden_size是从上一个节点传过来的向量的长度和输出的向量长度，num_layers是RNN的层数，
# batch_first是指输入的数据是否是batch在前，true的话是(batch, seq, feature)的形式，false的话是(seq, batch, feature)的形式

# input: (batch, seq_len, input_size),或者(batch, seq_len, input_size),根据batch_first的值来决定
input = torch.randn(5, 3, 5)
# h0: (num_layers * num_directions, batch, hidden_size) num_directions是指双向还是单向，1是单向，2是双向
h0 = torch.zeros(1, 5, 10)
output, hn = rnn(input, h0)
print(output.shape, hn.shape)# output是RNN的输出(y_hat)，hn是RNN的最后一个传向下一个节点的向量
# output: (batch, seq_len, num_directions * hidden_size) 或 (seq_len, batch, num_directions * hidden_size)，根据batch_first的值来决定
# h1: (num_layers * num_directions, batch, hidden_size)
# print输出torch.Size([5, 3, 10]) torch.Size([1, 5, 10])

rnn = nn.RNN(input_size=5, hidden_size=10, num_layers=2, batch_first=True)
input = torch.randn(5, 3, 5)
h0 = torch.zeros(2, 5, 10)
output, hn = rnn(input, h0)
print(output.shape, hn.shape) # print输出torch.Size([5, 3, 10]) torch.Size([2, 5, 10])

# 双向RNN
rnn = nn.RNN(input_size=5, hidden_size=10, num_layers=2, batch_first=True, bidirectional=True)
input = torch.randn(5, 3, 5)
h0 = torch.zeros(4, 5, 10)
output, hn = rnn(input, h0)
print(output.shape, hn.shape) # print输出torch.Size([5, 3, 20]) torch.Size([4, 5, 10])
# hn[0]是第一层正向的最后一个隐藏层，hn[1]是第一层反向的最后一个隐藏层，hn[2]是第二层正向的最后一个隐藏层，hn[3]是第二层反向的最后一个隐藏层
# output[x, y, 0:10]是第x个batch的第y个词的正向的输出，output[x, y, 10:20]是第x个batch的第y个词的反向的输出

# RNNCell layer(一步一步来)
rnn_cell = nn.RNNCell(input_size=5, hidden_size=10)# 只有一个返回值，就是RNN传向下一个节点的向量，要获得y_hat，需要自己将hx通过一个线性层，加上一个激活函数
# input_size是输入的一个词向量的长度，hidden_size是从上一个节点传过来的向量的长度和输出的向量长度
# input: (batch, seq_len, input_size)
input = torch.randn(5, 3, 5)
# h0: (batch, hidden_size)
h0 = torch.zeros(5, 10)
output = []
seq_len = 3
for i in range(seq_len):
    h0 = rnn_cell(input[:, i, :], h0)
    output.append(h0)
print(len(output), output[0].shape) # print输出3 torch.Size([5, 10])

# 两层RNNCell
cell1 = nn.RNNCell(input_size=5, hidden_size=10)
cell2 = nn.RNNCell(input_size=10, hidden_size=20)
input = torch.randn(5, 3, 5)
h01 = torch.zeros(5, 10)
h02 = torch.zeros(5, 20)
output = []
seq_len = 3
for i in range(seq_len):
    h01 = cell1(input[:, i, :], h01)
    h02 = cell2(h01, h02)
    output.append(h02)
print(len(output), output[0].shape) # print输出3 torch.Size([5, 20])


