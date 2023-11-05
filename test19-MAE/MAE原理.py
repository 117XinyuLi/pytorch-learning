# MAE：Masked AutoEncoder
# 网址：https://github.com/facebookresearch/mae
# 论文：https://arxiv.org/abs/2111.06377
# 使用了自监督学习

# 将图片转换为一个个块，然后遮住其中一部分(75%)，让网络学习重建遮住的部分
# 使用了非对称的encoder和decoder，encoder输入没被遮掩的部分，输出与遮住部分的embedding拼起来，输入decoder，输出重建的图片(这个过程叫pretrain)
# 同时，用这种方法可以让encoder更高效地学习到图片的特征，将不遮住图片的encoder接上分类器后可以用来做图像分类（可以只训练分类器：linear probing，也可以微调encoder和分类器：fine tune），
# 得到的ViT效果比原始ViT要好，linear probing的效果比fine tune的效果要差，linear probing更需要前面的层比较深，可以学到更多的特征
# 对于pretrain，做不做data augmentation都可以，影响不大，但是对于linear probing和fine tune，做data augmentation效果更好

# Masking:将图片划分为一个个没有重合的patch，随机挑选（75%）遮住，遮住部分大是为了消除图片的冗余性，不让网络轻易学习出来
# Encoder:ViT的结构，接受图片，进行patch embedding、position embedding，进行random shuffle(dim=1)，选取前25%，加上class token，输入到Transformer blocks中，输出patch
# Decoder:接受encoder的输出，线性变换到decoder的维数上，拼上mask token, 进行random shuffle的反操作(un-shuffle,dim=1)，将patch还原到原来的位置，
#         再加上class token, position embedding，输入到Transformer blocks中，输出为重建的patch，除去class token，再连接MLP，输出重建的图片
# Encoder用于提取图片的特征，Decoder用于重建图片，Decoder可以比较小，比较浅
# 这里pos embedding是按正余弦函数生成的定值
# Loss:MSE,输出图片和重建图片在遮住位置的MSE
# 将每个patch归一化后训练效果更好

# 具体实施：图片输入encoder，得到的输出输入decoder，输出重建的图片，计算MSE，进行反向传播




