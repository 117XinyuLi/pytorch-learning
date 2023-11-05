# AutoEncoder

# 无监督学习
# Why needed: Dimension Reduction(pre-processing) Generate new data...

# AutoEncoder: input => encoder => code(一般是降维) => decoder => output(一般是还原)
# loss: 如果衡量的是还原的误差，那么就是MSE，如果把输出的结果(各像素)看成是概率(0~1)(比如说灰度图片)，那么就是交叉熵，类似于分类问题
# 相对于PCA，AutoEncoder可以学习到非线性的特征，而且可以学习到非常复杂的特征，而PCA进行线性变换，只能学习到线性的特征

# AutoEncoder变种
# 1. Denoising AutoEncoder: 加入噪声，让网络学习去除噪声
#    input =(加入噪声)=> noise input => encoder => code => decoder => output
#    加入噪声，让网络学习更深层次的特征，增强泛化能力
# 2. Dropout AutoEncoder: 在训练过程中，加入Dropout
#    input => encoder(加入Dropout) => code => decoder(加入Dropout) => output
# 3. Adversarial AutoEncoder: 加入对抗网络，让网络学习到更复杂的特征
#    AutoEncoder中生成的code的分布会随着训练的进行而变化，而对抗网络的目的是让生成的code的分布不变
#    input(x) => encoder(θ) => code(z) => decoder(φ) => output(x)
#                               |(输入到对抗网络)
#                     某分布 => input => Discriminator => 0/1(0表示生成的code的分布和希望的分布不一样，1表示一样)
#    这里不给出更多的细节，后面会讲到GAN，这里只是简单的介绍一下
# 4. Variational AutoEncoder(VAE): 生成的code服从某种分布
#    input(x) => encoder(θ) => code(z) => decoder(φ) => output(x)
#    loss函数，以下公式[]表示右下标
#    loss[i](θ, φ) = -E[z~p[θ](z|x[i])](log(p[φ](x[i]|z))) + β*KL(p[θ](z|x[i])||p(z)), 其中KL(P||Q) = p(x)log(p(x)/q(x))dx从负无穷到正无穷的积分
#                  = reconstruction loss(输入输出的差异) + β*KL loss(生成的code的分布和希望的分布的差异)
#    第一部分为期望，在φ网络下，-log(p[φ](x[i]|z))是希望在z的前提下生成的x[i]的概率越大越好，其中z服从θ网络的分布，在输入x[i]的前提下，生成z
#                 工程上，可以用MSE，(output_x - input_x)^2来计算
#    第二部分为KL散度，衡量的是两个分布的差异，这里是衡量生成的z的分布和希望的z的分布(需要自行预设分布)的差异，两个分布越相似，KL越小
#       KL(p[θ](z|x[i])||p(z))这里希望p[θ](z|x[i])能够接近p(z)，KL小
#       若两分布均为一维高斯分布，KL散度可以用均值和方差来计算，KL(P||Q) = log(σq/σp) + (σp^2 + (μp - μq)^2)/(2σq^2) - 1/2
#       不加入KL散度的话，可能只会学习一些只能代表某些特定输入的特征，学习一个small variance的分布，而不是学习到更通用的特征
#    β用于控制两部分的平衡，一般β=1，KL loss比reconstruction loss小不少
#
#    Reparameterization Trick:
#    普通AutoEncoder的code(z)是与输入x一一对应的编码，而Variational AutoEncoder的code是一个概率分布，需要从这个分布中采样，得到编码，再输入到decoder中
#    但是直接从分布中采样这个操作是不可导的，不能backpropagation，可以令z = μ + σ*ϵ，其中ϵ为标准正态分布，从其中采样，这样就可以对μ和σ进行backpropagation了，而且z服从N(μ, σ^2)
#    比如，假设z是一个二维向量，那么encoder可以输出μ1和σ1，μ2和σ2，那么z1 = μ1 + σ1*ϵ; z2 = μ2 + σ2*ϵ，这样就可以对μ1和σ1，μ2和σ2进行backpropagation了
#    再将z1和z2输入到decoder中，就可以得到output_x了
#    计算KL散度时，比如，希望得到的(μ1,σ2)...(μn,σn)均是一个均值为0，方差为1的正态分布，那么对(μ1,σ2)...(μn,σn)分别和标准正态分布计算KL散度，然后求和
#    ...=> encoder(θ) => μ1, σ1, μ2, σ2 => z1 = μ1 + σ1*ϵ; z2 = μ2 + σ2*ϵ => decoder(φ) => output_x
#
#    AutoEncoder不是一个生成模型，而是一个编码模型(降维)，它的目的是将输入的x编码成一个code，然后再解码成一个output_x，这个output_x和输入的x是一样的
#    Variational AutoEncoder是一个生成模型，它的目的是将输入的x编码成一个分布，然后再由这个分布解码成一个output_x，这个output_x和输入的x不一定是一样的，但是它们是来自同一个分布的
#    重新生成的时候，VAE一开始会比AE更模糊，但是随着训练的进行，VAE会比AE更清晰
# 5.Convolutional AutoEncoder
#    与普通AutoEncoder的区别在于，encoder和decoder都是卷积神经网络，而不是全连接神经网络


