# ViT是Encoder only的，没有Decoder
# 只用于分类，需要的数据量比较大
# 这里只展示一层的ViT，实际上有多层叠加

import torch
import torch.nn as nn
import torch.nn.functional as F


# Step 1: Patch Embedding
def image2embedding_naive(image, patch_size, weight):
    # image: [batch_size, Ch, H, W]
    # patch_size: scalar
    # weight: [Ch*patch_size*patch_size, embedding_dim]
    # output: [batch_size,  H*W/patch_size/patch_size, embedding_dim]
    patch = F.unfold(image, kernel_size=patch_size, stride=patch_size)# [batch_size, Ch*patch_size*patch_size, H*W/patch_size/patch_size]
    patch = patch.transpose(1, 2)# [batch_size, H*W/patch_size/patch_size, Ch*patch_size*patch_size] 图片数，patch数，patch打平后的维度
    patch_embedding = torch.matmul(patch, weight)# [batch_size, H*W/patch_size/patch_size, embedding_dim] 图片数，patch数，embedding维度
    return patch_embedding


def image2embedding_conv(image, kernel, stride):
    # image: [batch_size, Ch, H, W]
    # kernel: [embedding_dim, Ch, patch_size, patch_size]
    # stride: scalar, patch_size
    # output: [batch_size,  H*W/patch_size/patch_size, embedding_dim]
    conv_out = F.conv2d(image, kernel, stride=stride, padding=0)# [bs, output_ch, output_h, output_w]
    bs, output_ch, output_h, output_w = conv_out.shape
    conv_out = conv_out.view(bs, output_ch, output_h*output_w).transpose(1, 2)# [bs, output_h*output_w, output_ch], 模仿nlp把channel放到最后
    return conv_out


# Step 2: Prepend a class token
def prepend_class_token(embedding):
    # embedding: [batch_size,  H*W/patch_size/patch_size, embedding_dim]
    # output: [batch_size,  H*W/patch_size/patch_size+1, embedding_dim]
    cls_token = nn.Parameter(torch.randn(embedding.shape[0], 1, embedding.shape[2])).to(embedding.device)# [batch_size, 1, embedding_dim]
    embedding = torch.cat([cls_token, embedding], dim=1)# [batch_size,  H*W/patch_size/patch_size+1, embedding_dim]
    return embedding


# Step 3: Positional Encoding
def positional_encoding(token_embedding, max_num_token):
    # token_embedding: [batch_size,  H*W/patch_size/patch_size+1, embedding_dim]
    # max_num_token: scalar, > H*W/patch_size/patch_size+1
    # output: [batch_size,  H*W/patch_size/patch_size+1, embedding_dim]
    position_embedding_table = nn.Parameter(torch.randn(max_num_token, token_embedding.shape[2])).to(token_embedding.device)
    #                                                                                          [max_num_token, embedding_dim]
    seq_len = token_embedding.shape[1]
    position_embedding = torch.tile(position_embedding_table[:seq_len], (token_embedding.shape[0], 1, 1))# [batch_size, seq_len, embedding_dim]
    # torch.tile()的作用是把一个tensor沿着某个维度复制多次，比如torch.tile(a, (2, 3))，就是把a沿着第0维复制2次，沿着第1维复制3次
    # 这里传入二维张量，指定有三个维度(a, b, c)，是把张量复制a次，每个张量沿着第0维复制b次，沿着第1维复制c次，然后把这些张量stack(dim=0)起来
    token_embedding = token_embedding + position_embedding# [batch_size,  seq_len, embedding_dim]
    return token_embedding


# Step 4: Pass embedding through transformer
def transformer_encoder(positional_encode, num_layers=6, num_heads=8):
    # positional_encode: [batch_size,  H*W/patch_size/patch_size+1, embedding_dim]
    # num_layers: scalar
    # num_heads: scalar
    # output: [batch_size,  H*W/patch_size/patch_size+1, embedding_dim]
    encoder_layer = nn.TransformerEncoderLayer(d_model=positional_encode.shape[2], nhead=num_heads).to(positional_encode.device)
    tf_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers).to(positional_encode.device)
    transformer_encode = tf_encoder(positional_encode).to(positional_encode.device)# [batch_size,  seq_len, embedding_dim]
    return transformer_encode


# Step 5: classification
def classification(transformer_encode, num_classes):
    # transformer_encode: [batch_size,  H*W/patch_size/patch_size+1, embedding_dim]
    # num_classes: scalar
    # output: [batch_size, num_classes]
    cls_token = transformer_encode[:, 0, :]# [batch_size, embedding_dim]
    linear_layer = nn.Linear(cls_token.shape[1], num_classes).to(cls_token.device)
    logits = linear_layer(cls_token).to(cls_token.device)# [batch_size, num_classes]
    return logits


class ViT(nn.Module):
    def __init__(self, image_size, patch_size, embedding_dim, max_num_token, num_classes, num_layers=6, num_heads=8, embedding_mode='conv'):
        super(ViT, self).__init__()
        self.ch, self.h, self.w = image_size
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.embedding_mode = embedding_mode

        if max_num_token < self.h * self.w / self.patch_size / self.patch_size + 1:
            raise ValueError('max_num_token should be larger than '+str(self.h * self.w / self.patch_size / self.patch_size + 1))

        self.max_num_token = max_num_token

        if self.embedding_mode == 'conv':
            self.kernel = nn.Parameter(torch.randn(embedding_dim, self.ch, patch_size, patch_size))
        elif self.embedding_mode == 'naive':
            self.weight = nn.Parameter(torch.randn(patch_size*patch_size*self.ch, embedding_dim))
        else:
            raise NotImplementedError

    def forward(self, image):
        # image: [batch_size, Ch, H, W]
        # output: [batch_size, num_classes]
        if self.embedding_mode == 'conv':
            embedding = image2embedding_conv(image, self.kernel, self.patch_size)
        elif self.embedding_mode == 'naive':
            embedding = image2embedding_naive(image, self.patch_size, self.weight)
        else:
            raise NotImplementedError
        embedding = prepend_class_token(embedding)
        embedding = positional_encoding(embedding, self.max_num_token)
        transformer_encode = transformer_encoder(embedding, self.num_layers, self.num_heads)
        logits = classification(transformer_encode, self.num_classes)
        return logits


if __name__ == '__main__':

    batchsz, ic, ih, iw = 2, 3, 8, 8
    model_dim = 8
    patch_size = 4
    patch_depth = patch_size ** 2 * ic
    x = torch.randn(batchsz, ic, ih, iw)

    """
    weight = torch.randn(patch_depth, model_dim)
    out = image2embedding_naive(x, patch_size, weight)
    print(out.shape)# [2, 4, 8]

    kernel = weight.transpose(0, 1).view(model_dim, ic, patch_size, patch_size)# output_ch, input_ch, kernel_h, kernel_w
    out = image2embedding_conv(x, kernel, stride=patch_size)
    print(out.shape)# [2, 4, 8]

    out = prepend_class_token(out)
    print(out.shape)# [2, 5, 8]
    out = positional_encoding(out, max_num_token=16)
    print(out.shape)# [2, 5, 8]

    out = transformer_encoder(out)
    print(out.shape)# [2, 5, 8]

    out = classification(out, num_classes=10)
    print(out.shape)# [2, 10]
    """

    model = ViT(image_size=(ic, ih, iw), patch_size=patch_size, embedding_dim=model_dim, max_num_token=16, num_classes=10, embedding_mode='conv')
    out = model(x)
    print(out.shape)# [2, 10]



