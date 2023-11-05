# embedding
import torch
import torch.nn as nn
from torchtext.vocab import GloVe

# embedding
word_to_ix = {"hello": 0, "world": 1}
embeds = nn.Embedding(2, 5)  # 2 words in vocab, 5 dimensional embeddings
lookup_tensor = torch.tensor([word_to_ix["hello"]], dtype=torch.long)
hello_embed = embeds(lookup_tensor)
print(hello_embed)
world_embed = embeds(torch.tensor([word_to_ix["world"]], dtype=torch.long))
print(world_embed)
# 以上的embedding是没有训练的，只是随机初始化的

# embedding with GloVe
cache_dir = '.vector_cache'# 缓存目录
glove = GloVe(name='6B', dim=300, cache=cache_dir)# 加载GloVe词向量 6B是名字(包含vector的文件的名称)，300是每个词向量的维度
print(glove.vectors.shape)# (400000, 300) 400000个单词，每个单词300维
print(glove.stoi['the']) # 0 is the index of the word 'the' in the vocabulary
print(glove['the'])# the vector of the word 'the'
print(glove.vectors[glove.stoi['the']])# the vector of the word 'the'




