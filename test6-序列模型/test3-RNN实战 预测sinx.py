# 预测正弦波形
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# 生成训练数据
num_time_steps = 100  # rnn time step / image height
start = np.random.randint(3, size=1)[0]  # 随机生成起始位置[0,3)
time_steps = np.linspace(start, start + 10, num_time_steps, dtype=np.float32)  # 生成100个数据
data = np.sin(time_steps)
data = data.reshape(num_time_steps, 1)
x_train = torch.from_numpy(time_steps[:-10]).float().view(1, -1, 1)
x_test = torch.from_numpy(time_steps[10:]).float().view(1, -1, 1)
y_train = torch.from_numpy(data[:-10]).float().view(1, -1, 1)  # 生成训练数据
y_test = torch.from_numpy(data[10:]).float().view(1, -1, 1)  # 生成测试数据


# 定义模型
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, num_layers=2):
        super(Net, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        # x (batch, seq_len, input_size)
        out, hidden = self.rnn(x, hidden)
        out = out.view(-1, self.hidden_size)
        out = self.fc(out)
        out = out.unsqueeze(dim=0)
        return out, hidden

    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        return hidden


# 训练模型
model = Net(input_size=1, hidden_size=16, output_size=1, num_layers=3)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
hidden = model.init_hidden(batch_size=1)
for epoch in range(6000):
    optimizer.zero_grad()
    output, hidden = model(x_train, hidden)
    hidden = hidden.detach()# 让hidden从计算图中分离出来，不用计算梯度
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print('Epoch: {}, Loss: {:.5f}'.format(epoch, loss.item()))

# 预测
model.eval()
hidden = model.init_hidden(batch_size=1)
test_output, _ = model(x_test, hidden)
data_predict = test_output.data.numpy()
dataY_plot = y_test.data.numpy()
data_predict = data_predict.reshape(num_time_steps - 10, 1)
dataY_plot = dataY_plot.reshape(num_time_steps - 10, 1)
plt.figure()
plt.plot(time_steps[10:], dataY_plot, 'r', label='original sin')
plt.plot(time_steps[10:], data_predict, 'b', label='predict sin')
plt.legend(loc='best')
plt.show()


