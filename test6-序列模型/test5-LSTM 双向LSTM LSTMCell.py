# LSTM LSTMCell
# 出现梯度消失，可以使用LSTM

import torch
import torch.nn as nn

# LSTM
LSTM = nn.LSTM(input_size=10, hidden_size=20, num_layers=2, batch_first=True, bidirectional=True)
# input和h0的shape类似RNN，比RNN多了一个c0，shape为(num_layers * num_directions, batch, hidden_size)，和h0一样
input = torch.randn(5, 3, 10)
h0 = torch.zeros(2*2, 5, 20)# 2*2是因为双向,有两层
c0 = torch.zeros(2*2, 5, 20)
output, (hn, cn) = LSTM(input, (h0, c0))# 也可以只传入input，不传入h0和c0，此时会自动初始化为0
print(output.shape, hn.shape, cn.shape)# torch.Size([5, 3, 40]) torch.Size([4, 5, 20]) torch.Size([4, 5, 20])
# 得到torch.Size([5, 3, 40])，5是batch_size，3是序列长度，40是hidden_size*2，因为是双向的

# LSTMCell
LSTMCell = nn.LSTMCell(input_size=10, hidden_size=20)# 返回hx和cx，hx可以通过线性层/softmax得到输出
# input和h0的shape类似RNN，比RNN多了一个c0，shape为(batch, hidden_size)，和h0一样
input = torch.randn(5, 3, 10)
h0 = torch.zeros(5, 20)
c0 = torch.zeros(5, 20)
output = []
seq_len = 3
for i in range(seq_len):
    h0, c0 = LSTMCell(input[:, i, :], (h0, c0))
    output.append(h0)
output = torch.stack(output, dim=1)
print(output.shape)# torch.Size([5, 3, 20])

# LSTMCell多层
LSTMCell1 = nn.LSTMCell(input_size=10, hidden_size=20)
LSTMCell2 = nn.LSTMCell(input_size=20, hidden_size=20)
input = torch.randn(5, 3, 10)
h01 = torch.zeros(5, 20)
c01 = torch.zeros(5, 20)
h02 = torch.zeros(5, 20)
c02 = torch.zeros(5, 20)
output = []
seq_len = 3
for i in range(seq_len):
    h01, c01 = LSTMCell1(input[:, i, :], (h01, c01))
    h02, c02 = LSTMCell2(h01, (h02, c02))
    output.append(h02)
output = torch.stack(output, dim=1)
print(output.shape)# torch.Size([5, 3, 20])









