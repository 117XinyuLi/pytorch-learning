# 情感分类实战
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext import data
from torchtext import datasets
import numpy as np

# 加载数据
TEXT = data.Field(tokenize='spacy', tokenizer_language='en_core_web_sm')
LABEL = data.LabelField(dtype=torch.float)
train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
print(f'Number of training examples: {len(train_data)}')# 25000
print(f'Number of testing examples: {len(test_data)}')# 25000
print(train_data.examples[0].text)
print(train_data.examples[0].label)# pos/neg

TEXT.build_vocab(train_data, max_size=10000, vectors='glove.6B.100d')
LABEL.build_vocab(train_data)

batchsz = 30
device = torch.device('cuda')
train_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, test_data),
    batch_size=batchsz,
    device=device
)

for batch in train_iterator:
    print(batch.text)
    print(batch.label)# shape:[b], 0/1
    break

# 模型
class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=2, bidirectional=True, dropout=0.5)
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        output, (hidden, cell) = self.rnn(embedded)
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        hidden = self.dropout(hidden)
        out = self.fc(hidden)
        return out


# 训练
# embedding初始化
rnn = RNN(len(TEXT.vocab), 100, 256)
pretrained_embeddings = TEXT.vocab.vectors
rnn.embedding.weight.data.copy_(pretrained_embeddings)

optimizer = optim.Adam(rnn.parameters(), lr=1e-3)
criterion = nn.BCEWithLogitsLoss().to(device)
rnn.to(device)


# 评估函数
def binary_acc(preds, y):
    preds = torch.round(torch.sigmoid(preds))
    correct = torch.eq(preds, y).float()
    acc = correct.sum() / len(correct)
    return acc


# 训练函数
def train(rnn, iterator, optimizer, criterion):
    avg_acc = []
    rnn.train()

    for i, batch in enumerate(iterator):

        # [seq, b] => [b, 1] => [b]
        pred = rnn(batch.text).squeeze(1)

        loss = criterion(pred, batch.label)
        acc = binary_acc(pred, batch.label).item()
        avg_acc.append(acc)

        optimizer.zero_grad()
        loss.backward()

        for p in rnn.parameters():
            torch.nn.utils.clip_grad_norm_(p, 1)

        optimizer.step()

        if i % 10 == 0:
            print(i, acc)

    avg_acc = np.array(avg_acc).mean()
    print('avg acc:', avg_acc)


# 测试函数
def eval(rnn, iterator, criteon):
    avg_acc = []

    rnn.eval()

    with torch.no_grad():
        for batch in iterator:
            # [b, 1] => [b]
            pred = rnn(batch.text).squeeze(1)

            loss = criteon(pred, batch.label)

            acc = binary_acc(pred, batch.label).item()
            avg_acc.append(acc)

    avg_acc = np.array(avg_acc).mean()

    print('>>test:', avg_acc)


for epoch in range(10):
    eval(rnn, test_iterator, criterion)
    train(rnn, train_iterator, optimizer, criterion)

