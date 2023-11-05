# Score Diffusion是Stable Diffusion的基础
# Score：概率密度p(x)log值对x的导数(Score(x))
# Score network：训练网络能够输出Score, 输出为S(x)
# Langevin Dynamics：采样算法，基于Score进行采样生成新的样本
#          设定epsilon>0, 采样次数T, 初始样本x0, 从某个先验分布π(x)中采样(x0~π(x)), 则x[t+1]=x[t]+epsilon*Score(x[t])/2+sqrt(epsilon)*N(0,1)
#          epsilon->0, T->∞, x[t]就是所需要的分布p(x)的采样
#          将公式中的Score替换为Score network的输出，就是通过Score-based model的基本思想
# Score Matching：使Score network的输出与Score相等，最小化E[||Score(x)-S(x)||^2]
#          等价于最小化E[tr(▽S(x))+0.5||S(x)||^2)], 其中tr(▽S(x))为S雅各比矩阵的迹, tr(▽S(x))比较难求解, 换为使用Denoising Score Matching
# Denoising Score Matching：对数据样本加预先设定的噪声，变为q(x_bar|x)然后估计加噪后的Score, 其中q(x_bar)=∫q(x_bar|x)p(x)dx
#          上面的目标函数就等价于0.5E(q(x_bar|x)*p(x))[||Score(x_bar|x)-S(x_bar)||^2], 其中E后面的小括号为E的右下标,
#          也就是相当于网络的输出和加噪后分布的Score要尽可能的接近, 当噪声足够小时, S(x)就是Score Matching所需要的结果
# 以上过程在低密度的区域Score计算不准确, 为了解决这个问题, 提出了Noise Conditional Score Network(NCSN)

# NCSN：Noise Conditional Score Network
# 对数据进行加噪，使得低密度区域更少，从而提高Score的准确性
# 如何加噪：加得少的话，低密度区域还是很多，加得多的话，破坏了原始数据的分布，
#         一种平衡的方案：加不同量级的噪声(σ1~n为系数小于一的等比数列)，使用一个网络来估计不同噪声下的Score
# 训练完NCSN后，可以使用Langevin Dynamics进行采样，生成新的样本，采样时一开始从噪声比较大的开始，然后逐渐减小噪声，直到噪声为0，这样生成的样本就是从原始分布的
# NCSN不同于上面的Score network，NCSN以上面的x和加入噪声的放上σ作为输入，输出为S(x,σ)，输入输出同维度
# 训练NCSN：使用Denoising Score Matching的思想，设定噪声q(x_bar|x)=N(x_bar|x,σ^2I), Score(x_bar|x)=-(x_bar-x)/σ^2
#         优化目标函数为L(θ,σ)=0.5E(p(x))E(x_bar~N(x,σ^2I))[||S(x_bar,σ)+(x_bar-x)/σ^2||^2]
#         x就是输入的x，x_bar就是加入噪声的x，σ就是噪声的系数，S(x_bar,σ)就是网络的输出，θ就是网络的参数
#         对于L个不同的噪声系数σ[i]，Loss(θ,σ)=1/L*(∑λ(σ[i])*L(θ,σ[i])), NCSN足够强的话就可以预测任意噪声下的Score
#         经验上λ(σ)=σ^2
# 使用退火的郎之万动力学进行采样：
#         Require: σ，epsilon，T
#         从随机分布中采样x[0]
#         for i=1 to L
#            σ[i]=epsilon*σ[i]^2/σ[L]^2
#            for t=1 to T
#               z=N(0,1)
#               x[t]=x[t-1]+σ[i]/2*S(x[t-1],σ[i])+z*sqrt(σ[i])
#            x[0]=x[T]
#         return x[T]
# 以上两种过程主要体现逆扩散过程，可用SDE来统一

# Score-based Generative Modeling with Stochastic Differential Equations(SDE)
# DDPM和NCSN都是SDE的特殊情况，SDE是一种更通用的方法
# 在扩散过程中，不断对数据加噪，变成一个噪声分布，从x[0]开始，不断的加噪，直到x[T]
# 如果有无限步加噪，那么x可以用一个微分方程来表示，这个微分方程就是SDE dx=f(x,t)dt+g(t)dW f是漂移系数，g是扩散系数，W是布朗运动的量
# 逆过程：从x[T]开始，不断的减噪，直到x[0] dx=(f(x,t)-g^2(t)▽log(p_t(x)))dt+g(t)dW，在某些情况下，可用欧拉方法等数值方法来求解
# 所以只要知道每个t时刻的x的Score，就可以通过SDE来生成x
# 所以需要训练时间相关的网络S(x,t)
# 需要最小化E(t~U(0,T))[λ(t)E(x[0]~p_0(x))E(x~p_t(x)|x[0])[||S(x,t)-▽log(p_t(x|x[0]))||^2]]
# x[0]是训练样本，x[t]在f(x,t)为仿射变换时，可以用重参数技巧获得λ(t)=1/E[||▽log(p_t(x|x[0]))||^2], 如果为高斯分布，这就是高斯分布的方差

# 设计分数网络
# 需要输入输出为同维度
# 建议用U-Net结构
# 在SDE中可以使用高斯随机特征将时间信息加入网络中
#     w~N(0,s^2I) 这里w是一个向量，s是一个标量
#     将[sin(2πwt);cos(2πwt)]作为输入的一部分
# U-Net的输出可以乘以1/sqrt(E[||▽log(p_t(x|x[0]))||^2])让输出的L2范数和分数的L2范数接近
# 可以使用EMA(指数移动平均,exponential moving average)来进采样加速训练










