# 对抗神经网络(GAN)
# 生成器(Generator)和判别器(Discriminator)
# 生成器生成假的图片，判别器判断图片是真是假
# 生成器和判别器是对抗的，生成器越来越好，判别器越来越差，直到判别器无法判断图片是真是假
#                                               Training set(满足分布Pr(x)) =>NN
#                                                                           ||(输入)
# Random noise(满足分布Pz(z)) => Generator(G) => Fake image(满足分布Pg(x)) => Discriminator(D) => 0(Fake)/1(Real)(0.5表示分辨不出来，纳什均衡)
# 训好的生成器就可以输入随机噪声生成假的图片
#
# How to train GAN?
# []表示右下标
# min_G(max_D(Loss(D, G))) = E[x~Pr(x)](log(D(x))) + E[z~Pz(z)](log(1 - D(G(z))))
#     (V(D,G))             = E[x~Pr(x)](log(D(x))) + E[x~Pg(x)](log(1 - D(x)))
# 对于D来说，想要最大化Loss(D, G)，就是希望D(x)尽可能接近1，D(G(z))尽可能接近0，把真的判为1，假的判为0
# 对于G来说，想要最小化Loss(D, G)，就是希望D(G(z))尽可能接近1，希望D认为G生成的图片是真的
# Where will D go(fixed G)?
# V(D,G) =( Pr(x)log(D(x)) + Pg(x)log(1 - D(x)) ) dx 对x积分(用期望的公式)
# 令dV/dD = 0，得到D的最优解, D* = Pr(x)/(Pr(x) + Pg(x))∈[0, 1]
# Where will G go(fixed D)?
# JS散度(Jensen-Shannon divergence) D_JS(P||Q) = 1/2 * D_KL(P||M) + 1/2 * D_KL(Q||M) 其中M = 1/2 * (P + Q)
# D_JS(Pr(x)||Pg(x)) = 1/2 * D_KL(Pr(x)||M) + 1/2 * D_KL(Pg(x)||M)
#                    = 1/2 * (log2+(Pr(x)log(Pr(x)/(Pr(x)+Pg(x))))dx对x积分 + log2+(Pg(x)log(Pg(x)/(Pr(x)+Pg(x))))dx对x积分)
#                    = 1/2 * (log4 + L(G,D*))
#            L(G,D*) = 2 * D_JS(Pr(x)||Pg(x)) - 2log2
# 令dL/dG = 0，得到G的最优解, 此时D* = 1/2，Pr(x) = Pg(x),从数学上推导出了，最终D和G都会收敛到纳什均衡
#
# DC-GAN(Deep Convolutional GAN)
# 用到transposed convolution layer，可以把图片放大
#
# 出现的问题：Training Stability
# JS散度的缺陷：如果两分布没有交叉，KL散度会变成无穷大，JS散度会恒等于log2，梯度消失无法训练
# 所以GAN该开始训练的时候会出现训练不起来
#
# EM距离(Earth Mover distance)：把两个分布看成是两堆土，把一堆土变成另一堆土的最小代价，也叫W距离Wasserstein Distance
# 设Average distance of a plan γ 为 B(γ) = ∑γ(xp,xq)||xp - xq||(对所有的xp, xq求和)
# EM距离W(P,Q) = min_γ B(γ)
# 用EM距离代替JS散度，可以解决Training Stability的问题
#
# WGAN(Wasserstein GAN) 用EM距离代替JS散度
# WGAN的训练方法：Loss[i](D, G) = f(x[i]) - f(G(z[i]))
# f需要用一个网络获得(GAN中的f就是log(D(x))，即利用JS散度)
# f需要满足Lipschitz条件，即|f(x) - f(y)| <= |x - y|，这样可以保证函数比较平滑，不会出现梯度消失的问题
# 可以使用weight clipping来满足Lipschitz条件，即把D的权重限制在[-c, c]之间(但是效果不好，约束了D的表达能力)
# WGAN-GP(WGAN with Gradient Penalty)：在Loss[i](D, G)的基础上加上一个梯度惩罚项，使得f更加平滑
# Loss = E[x'~Pg(x)](f(x')) - E[x''~Pr(x)](f(x'')) + λE[x~P(x)](||∇f(x)|| - 1)^2, 其中x=tx' + (1-t)x''，0<=t<=1,为x'和x''的线性插值点
# 这样就使得f的梯度在1附近，更加平滑，训练更稳定
# 可以在训练时打印出W距离，会发现W距离在不断减小，说明D和G都在收敛
