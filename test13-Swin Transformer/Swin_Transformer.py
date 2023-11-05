# 图片分为每个patch，比如3*32*32的图片，分成4*4的patch，每个patch的大小为4*4*3，一共64个patch [8, 8, 48]
# patch通过MLP进行embedding，得到C个通道的embedding，一共有64个patch [8, 8, 48] -> [8, 8, C]
# 通过Swin Transformer Block、MLP进行编码，得到C个通道的编码，一共有64个patch [8, 8, C] -> [8, 8, C]
# 通过patch merge，将64个patch的合并 [8, 8, C] -> [4, 4, C] 通过Swin Transformer Block、MLP [4, 4, C] -> [4, 4, 2C]
# 重复上一步
# 以上示例patch占用两个维度，代码中占用一个
# 通过MLP进行分类

# Swin Transformer Block中有LN、MLP、W-MSA(attention)、SW-MSA(attention, shift window)
# 使用了shift window, 加强不同patch之间的关系，一个window中有多个patch

import torch
import torch.nn as nn
import torch.nn.functional as F

# 注意：本代码中大量使用整除，开根取整，建议图片格式为(2, 2^n, 2^n)，shift window的方法很巧妙，可以学习


# 1. patch embedding
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


# 2. Multi-Head Attention
class MutiHeadSelfAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super(MutiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads

        self.proj_linear_layer = nn.Linear(embedding_dim, embedding_dim*3)
        self.final_linear_layer = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, image, additive_mask=None):
        # image: [batch_size, seq_len, embedding_dim]
        # additive_mask: [batch_size, seq_len, seq_len]
        # output: atten: [bs*num_heads, seq_len, seq_len], output: [bs, seq_len, embedding_dim]
        bs, seq_len, embedding_dim = image.shape
        head_dim = embedding_dim // self.num_heads

        q, k, v = self.proj_linear_layer(image).chunk(3, dim=-1)
        # [bs, seq_len, embedding_dim*3] -> [bs, seq_len, embedding_dim], [bs, seq_len, embedding_dim], [bs, seq_len, embedding_dim]

        q = q.view(bs, seq_len, self.num_heads, head_dim).transpose(1, 2)# [bs, num_heads, seq_len, head_dim]
        q = q.reshape(bs*self.num_heads, seq_len, head_dim)# [bs*num_heads, seq_len, head_dim]

        k = k.view(bs, seq_len, self.num_heads, head_dim).transpose(1, 2)# [bs, num_heads, seq_len, head_dim]
        k = k.reshape(bs*self.num_heads, seq_len, head_dim)# [bs*num_heads, seq_len, head_dim]

        v = v.view(bs, seq_len, self.num_heads, head_dim).transpose(1, 2)# [bs, num_heads, seq_len, head_dim]
        v = v.reshape(bs*self.num_heads, seq_len, head_dim)# [bs*num_heads, seq_len, head_dim]

        # q, k, v: [bs*num_heads, seq_len, head_dim]

        if additive_mask is None:
            attn_prob = F.softmax(torch.bmm(q, k.transpose(1, 2)) / (head_dim ** 0.5), dim=-1)# [bs*num_heads, seq_len, seq_len]
            # torch.bmm是矩阵乘法

        else:
            attn_mask = additive_mask.tile(self.num_heads, 1, 1)# [bs*num_heads, seq_len, seq_len]
            attn_prob = F.softmax(torch.bmm(q, k.transpose(1, 2)) / (head_dim ** 0.5) + attn_mask, dim=-1)# [bs*num_heads, seq_len, seq_len]

        output = torch.bmm(attn_prob, v)# [bs*num_heads, seq_len, head_dim]
        output = output.reshape(bs, self.num_heads, seq_len, head_dim).transpose(1, 2)# [bs, seq_len, num_heads, head_dim]
        output = output.reshape(bs, seq_len, embedding_dim)# [bs, seq_len, embedding_dim]

        output = self.final_linear_layer(output)# [bs, seq_len, embedding_dim]
        return attn_prob, output


# 3. window MHSA
def window_muti_head_self_attention(patch_embedding, mhsa, window_size=4):
    # patch_embedding: [batch_size, num_patch, embedding_dim]
    # mhsa: MutiHeadSelfAttention
    # window_size: scalar
    # num_head: scalar
    # output: [batch_size, num_patch/window_size/window_size, window_size*window_size, embedding_dim]
    num_patch_in_window = window_size * window_size
    bs, num_patch, embedding_dim = patch_embedding.shape
    image_height = image_width = int(num_patch ** 0.5)

    patch_embedding = patch_embedding.transpose(1, 2)# [bs, embedding_dim, num_patch]
    patch = patch_embedding.reshape(bs, embedding_dim, image_height, image_width)# [bs, embedding_dim, image_height, image_width]
    window = F.unfold(patch, kernel_size=window_size,
                      stride=window_size).transpose(1, 2).to(patch_embedding.device)
    # [bs, num_patch/num_patch_in_window, embedding_dim*num_patch_in_window]

    bs, num_window, patch_depth_in_window = window.shape
    window = window.reshape(bs*num_window, embedding_dim, num_patch_in_window)# [bs*num_window, embedding_dim, num_patch_in_window]
    window = window.transpose(1, 2)# [bs*num_window, num_patch_in_window, embedding_dim]

    attn_prob, output = mhsa(window)
    # [bs*num_window, num_patch_in_window, num_patch_in_window], [bs*num_window, num_patch_in_window, embedding_dim]

    output = output.reshape(bs, num_window, num_patch_in_window, embedding_dim)
    # [bs, num_window, num_patch_in_window, embedding_dim]
    return output


# 4. Shift window MHSA
def window2image(msa_output):
    # msa_output: [batch_size, num_patch/window_size/window_size, window_size*window_size, embedding_dim]
    # output: [batch_size, embedding_dim, image_height, image_width]
    #         image_height = image_width = int(num_patch ** 0.5)
    bs, num_window, num_patch_in_window, embedding_dim = msa_output.shape
    window_size = int(num_patch_in_window ** 0.5)
    image_height = int(num_window ** 0.5)*window_size
    image_width = image_height

    msa_output = msa_output.reshape(bs, int(num_window ** 0.5), int(num_window ** 0.5), window_size, window_size, embedding_dim)
    msa_output = msa_output.transpose(2, 3)# [bs, num_window**0.5, window_size, num_window**0.5, window_size, embedding_dim]
    image = msa_output.reshape(bs, image_height*image_width, embedding_dim)

    image = image.transpose(1, 2).reshape(bs, embedding_dim, image_height, image_width)
    # [bs, embedding_dim, image_height, image_width]
    return image


def build_mask_for_shift_window(bs, image_height, image_width, window_size, device):
    # bs: scalar
    # image_height: scalar
    # image_width: scalar
    # window_size: scalar
    # output: [bs*num_window, num_patch_in_window, num_patch_in_window]
    index_matrix = torch.zeros(image_height, image_width).to(device)
    for i in range(image_height):
        for j in range(image_width):
            row_times = (i + window_size//2) // window_size
            col_times = (j + window_size//2) // window_size
            index_matrix[i, j] = row_times * (image_height//window_size) + col_times + 1

    rolled_index_matrix = torch.roll(index_matrix, shifts=(-window_size//2, -window_size//2), dims=(0, 1))
    rolled_index_matrix = rolled_index_matrix.unsqueeze(0).unsqueeze(0)# [1, 1, image_height, image_width]
    c = F.unfold(rolled_index_matrix, kernel_size=window_size, stride=window_size).transpose(1, 2)
    # [1, num_window, window_size*window_size]
    c = torch.tile(c, (bs, 1, 1))# [bs, num_window, window_size*window_size]

    bs, num_window, num_patch_in_window = c.shape

    c1 = c.unsqueeze(-1)# [bs, num_window, num_patch_in_window, 1]
    c2 = (c1-c1.transpose(-1, -2)) == 0# [bs, num_window, num_patch_in_window, num_patch_in_window]
    # c2[i, j, k, l] = 1 if c1[i, j, k, 0] == c1[i, j, l, 0], 即第j个window的第k个patch和第l个patch(同一类patch)在同一个window中
    valid_matrix = c2.to(torch.float32)
    additive_mask = (1.0 - valid_matrix) * (-1e9)# [bs, num_window, num_patch_in_window, num_patch_in_window]
    # 将不在同一个window中的patch的attention置为-inf，使其在softmax中不会被计算

    additive_mask = additive_mask.reshape(bs*num_window, num_patch_in_window, num_patch_in_window)
    # [bs*num_window, num_patch_in_window, num_patch_in_window]
    return additive_mask


def shift_window(w_msa_output, window_size, shift_size, generate_mask=False):
    # 不直接计算移动窗口，利用roll将相同的窗口尽量移动到一起，然后通过mask确定哪些是需要计算的，不然要计算更多的窗口，这样计算和不移动的窗口数量一样
    # w_msa_output: [batch_size, num_patch/window_size/window_size, window_size*window_size, embedding_dim]
    # window_size: scalar
    # shift_size: scalar
    # generate_mask: bool
    # output: [batch_size, num_patch/window_size/window_size, window_size*window_size, embedding_dim]
    #         None/[bs*num_window, num_patch_in_window, num_patch_in_window]
    bs, num_window, num_patch_in_window, embedding_dim = w_msa_output.shape
    w_msa_output = window2image(w_msa_output)
    bs, embedding_dim, image_height, image_width = w_msa_output.shape
    roll_w_msa_output = torch.roll(w_msa_output, shifts=(shift_size, shift_size), dims=(2, 3))# 图片向左上/右下移动,移出的部分填到另一边
    # [bs, embedding_dim, image_height, image_width]
    shift_w_msa_input = roll_w_msa_output.reshape(bs, embedding_dim,
                                                  int(num_window**0.5),
                                                  window_size,
                                                  int(num_window**0.5),
                                                  window_size)
    shift_w_msa_input = shift_w_msa_input.transpose(3, 4)# [bs, embedding_dim, num_window**0.5, num_window**0.5, window_size, window_size]
    shift_w_msa_input = shift_w_msa_input.reshape(bs, embedding_dim, num_window * num_patch_in_window)
    shift_w_msa_input = shift_w_msa_input.transpose(1, 2)# [bs, num_window*num_patch_in_window, embedding_dim]
    shift_w_msa_input = shift_w_msa_input.reshape(bs, num_window, num_patch_in_window, embedding_dim)

    if generate_mask:
        additive_mask = build_mask_for_shift_window(bs, image_height, image_width, window_size, w_msa_output.device)
    else:
        additive_mask = None

    return shift_w_msa_input, additive_mask


def shift_window_muti_head_self_attention(w_msa_output, mhsa, window_size=4):
    # w_msa_output: [batch_size, num_patch/window_size/window_size, window_size*window_size, embedding_dim]
    # mhsa: MultiHeadSelfAttention
    # window_size: scalar
    # numhead: scalar
    # output: [batch_size, num_patch/window_size/window_size, window_size*window_size, embedding_dim]
    bs, num_window, num_patch_in_window, embedding_dim = w_msa_output.shape
    shift_w_msa_input, additive_mask = shift_window(w_msa_output, window_size, shift_size=-window_size//2, generate_mask=True)# 移动窗口

    shift_w_msa_input = shift_w_msa_input.reshape(bs*num_window, num_patch_in_window, embedding_dim)

    atten_prob, output = mhsa(shift_w_msa_input, additive_mask=additive_mask)

    output = output.reshape(bs, num_window, num_patch_in_window, embedding_dim)

    output, _ = shift_window(output, window_size, shift_size=window_size//2, generate_mask=False)# 将移动的窗口移回来

    return output


# 5. patch merging
class PatchMerging(nn.Module):
    def __init__(self, embedding_dim, merge_size, output_depth_scale=0.5):
        # merge_size: scalar 将多少个patch合并为一个patch
        super(PatchMerging, self).__init__()
        self.merge_size = merge_size
        self.proj_layer = nn.Linear(
            embedding_dim * merge_size * merge_size,
            int(embedding_dim * merge_size * merge_size * output_depth_scale)
        )

    def forward(self, image):
        # x: [batch_size, num_window, num_patch_in_window, embedding_dim]
        # output: [batch_size, num_patch/merge_size/merge_size, int(embedding_dim * merge_size * merge_size * output_depth_scale)]
        bs, num_window, num_patch_in_window, embedding_dim = image.shape
        window_size = int(num_patch_in_window**0.5)

        image = window2image(image)
        # [bs, embedding_dim, image_height, image_width]

        merged_window = F.unfold(image, kernel_size=(self.merge_size, self.merge_size),
                                 stride=(self.merge_size, self.merge_size)).transpose(1, 2)
        # [bs, num_patch/merge_size/merge_size, embedding_dim*merge_size*merge_size]

        merged_window = self.proj_layer(merged_window)

        return merged_window


# 6. Swin Transformer Block
class SwinTransformerBlock(nn.Module):
    def __init__(self, embedding_dim, window_size=4, numhead=2):
        super(SwinTransformerBlock, self).__init__()
        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)
        self.layer_norm3 = nn.LayerNorm(embedding_dim)
        self.layer_norm4 = nn.LayerNorm(embedding_dim)

        self.wsma_mlp1 = nn.Linear(embedding_dim, 4*embedding_dim)
        self.wsma_mlp2 = nn.Linear(4*embedding_dim, embedding_dim)
        self.swsma_mlp1 = nn.Linear(embedding_dim, 4*embedding_dim)
        self.swsma_mlp2 = nn.Linear(4*embedding_dim, embedding_dim)

        self.mhsa1 = MutiHeadSelfAttention(embedding_dim, numhead)
        self.mhsa2 = MutiHeadSelfAttention(embedding_dim, numhead)

        self.window_size = window_size
        self.numhead = numhead

    def forward(self, image):
        # image: [batch_size, num_patch, embedding_dim]
        # output: [batch_size, num_window, num_patch_in_window, embedding_dim]
        bs, num_patch, embedding_dim = image.shape

        input1 = self.layer_norm1(image)
        w_msa_output = window_muti_head_self_attention(input1, self.mhsa1, self.window_size)
        bs, num_window, num_patch_in_window, embedding_dim = w_msa_output.shape
        w_msa_output = image + w_msa_output.reshape(bs, num_patch, embedding_dim)
        output1 = self.wsma_mlp2(self.wsma_mlp1(self.layer_norm2(w_msa_output)))
        output1 += w_msa_output

        input2 = self.layer_norm3(output1)
        input2 = input2.reshape(bs, num_window, num_patch_in_window, embedding_dim)
        sw_msa_output = shift_window_muti_head_self_attention(input2, self.mhsa2, self.window_size)
        output2 = self.swsma_mlp2(self.swsma_mlp1(self.layer_norm4(sw_msa_output)))
        output2 += sw_msa_output

        output2 = output2.reshape(bs, num_window, num_patch_in_window, embedding_dim)

        return output2


# 7. Swin Transformer
class SwinTransformer(nn.Module):
    def __init__(self, input_image_channel=3, patch_size=4, model_dim_C=8, window_size=4, numhead=2, merge_size=2, num_classes=10, embedding_mode='conv'):
        super(SwinTransformer, self).__init__()
        patch_depth = input_image_channel * patch_size * patch_size
        self.patch_size = patch_size
        self.model_dim_C = model_dim_C
        self.num_classes = num_classes

        self.block1 = SwinTransformerBlock(model_dim_C, window_size, numhead)
        self.block2 = SwinTransformerBlock(model_dim_C*2, window_size, numhead)
        self.block3 = SwinTransformerBlock(model_dim_C*4, window_size, numhead)
        self.block4 = SwinTransformerBlock(model_dim_C*8, window_size, numhead)

        self.patch_merging1 = PatchMerging(model_dim_C, merge_size)
        self.patch_merging2 = PatchMerging(model_dim_C*2, merge_size)
        self.patch_merging3 = PatchMerging(model_dim_C*4, merge_size)

        self.fc = nn.Linear(model_dim_C*8, num_classes)

        self.embedding_mode = embedding_mode
        if self.embedding_mode == 'conv':
            self.kernel = nn.Parameter(torch.randn(model_dim_C, 3, patch_size, patch_size))
        elif self.embedding_mode == 'naive':
            self.patch_embedding_weight = nn.Parameter(torch.randn(patch_depth, model_dim_C))
        else:
            raise NotImplementedError

    def forward(self, x):
        if self.embedding_mode == 'conv':
            patch_embedding = image2embedding_conv(x, self.kernel, self.patch_size)
        elif self.embedding_mode == 'naive':
            patch_embedding = image2embedding_naive(x, self.patch_embedding_weight, self.patch_size)
        else:
            raise NotImplementedError

        sw_msa_output = self.block1(patch_embedding)
        sw_msa_output = self.block2(self.patch_merging1(sw_msa_output))
        sw_msa_output = self.block3(self.patch_merging2(sw_msa_output))
        sw_msa_output = self.block4(self.patch_merging3(sw_msa_output))

        bs, num_window, num_patch_in_window, embedding_dim = sw_msa_output.shape
        sw_msa_output = sw_msa_output.reshape(bs, -1, embedding_dim)

        pool_output = torch.mean(sw_msa_output, dim=1)
        logits = self.fc(pool_output)

        return logits


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SwinTransformer().to(device)
    x = torch.randn(2, 3, 256, 256).to(device)
    y = model(x)
    print(y.shape)

    # torch.Size([2, 10])

