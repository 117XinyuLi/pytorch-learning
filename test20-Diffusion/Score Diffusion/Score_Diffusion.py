# 这里使用SDE，没有conditioning的网络
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import functools
from copy import deepcopy
import tqdm
from scipy import integrate
from torchvision.utils import make_grid
import time
import matplotlib.pyplot as plt


class TimeEncoding(nn.Module):
    """用于对时间信息进行编码"""

    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        self.w = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.w[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Dense(nn.Module):
    """Dense layer"""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.dense(x)[..., None, None]  # 扩充维度


class ScoreNet(nn.Module):
    """基于U-Net的Score网络"""

    def __init__(self, marginal_prob_std, channels=[32, 64, 128, 256], embed_dim=256):
        super().__init__()
        # Embedding
        self.embed = nn.Sequential(
            TimeEncoding(embed_dim=embed_dim),
            nn.Linear(embed_dim, embed_dim))

        # U-Net编码器部分
        self.conv1 = nn.Conv2d(1, channels[0], 3, stride=1, bias=False)
        self.dense1 = Dense(embed_dim, channels[0])
        self.gnorm1 = nn.GroupNorm(4, num_channels=channels[0])
        self.conv2 = nn.Conv2d(channels[0], channels[1], 3, stride=2, bias=False)
        self.dense2 = Dense(embed_dim, channels[1])
        self.gnorm2 = nn.GroupNorm(32, num_channels=channels[1])
        self.conv3 = nn.Conv2d(channels[1], channels[2], 3, stride=2, bias=False)
        self.dense3 = Dense(embed_dim, channels[2])
        self.gnorm3 = nn.GroupNorm(32, num_channels=channels[2])
        self.conv4 = nn.Conv2d(channels[2], channels[3], 3, stride=2, bias=False)
        self.dense4 = Dense(embed_dim, channels[3])
        self.gnorm4 = nn.GroupNorm(32, num_channels=channels[3])

        # U-Net解码器部分
        self.tconv4 = nn.ConvTranspose2d(channels[3], channels[2], 3, stride=2, bias=False)
        self.dense5 = Dense(embed_dim, channels[2])
        self.tgnorm4 = nn.GroupNorm(32, num_channels=channels[2])
        self.tconv3 = nn.ConvTranspose2d(channels[2] + channels[2], channels[1], 3, stride=2, bias=False,
                                         output_padding=1)
        self.dense6 = Dense(embed_dim, channels[1])
        self.tgnorm3 = nn.GroupNorm(32, num_channels=channels[1])
        self.tconv2 = nn.ConvTranspose2d(channels[1] + channels[1], channels[0], 3, stride=2, bias=False,
                                         output_padding=1)
        self.dense7 = Dense(embed_dim, channels[0])
        self.tgnorm2 = nn.GroupNorm(32, num_channels=channels[0])

        self.act = lambda x: x * torch.sigmoid(x)

        self.tconv1 = nn.ConvTranspose2d(channels[0] + channels[0], 1, 3, stride=1)
        self.marginal_prob_std = marginal_prob_std

    def forward(self, x, t):
        # Embedding
        embed = self.act(self.embed(t))

        # U-Net编码器部分
        h1 = self.conv1(x)
        h1 += self.dense1(embed)  # 加上时间信息
        h1 = self.gnorm1(h1)
        h1 = self.act(h1)
        h2 = self.conv2(h1)
        h2 += self.dense2(embed)  # 加上时间信息
        h2 = self.gnorm2(h2)
        h2 = self.act(h2)
        h3 = self.conv3(h2)
        h3 += self.dense3(embed)  # 加上时间信息
        h3 = self.gnorm3(h3)
        h3 = self.act(h3)
        h4 = self.conv4(h3)
        h4 += self.dense4(embed)  # 加上时间信息
        h4 = self.gnorm4(h4)
        h4 = self.act(h4)

        # U-Net解码器部分
        h = self.tconv4(h4)
        h += self.dense5(embed)  # 加上时间信息
        h = self.tgnorm4(h)
        h = self.act(h)
        h = self.tconv3(torch.cat([h, h3], dim=1))
        h += self.dense6(embed)  # 加上时间信息
        h = self.tgnorm3(h)
        h = self.act(h)
        h = self.tconv2(torch.cat([h, h2], dim=1))
        h += self.dense7(embed)  # 加上时间信息
        h = self.tgnorm2(h)
        h = self.act(h)
        h = self.tconv1(torch.cat([h, h1], dim=1))

        # L2范数逼近
        h = h / self.marginal_prob_std(t)[:, None, None, None]

        return h


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 这定义扩散过程dx=σ^tdw t∈[0,1] 漂移系数为0
def marginal_prob_std(t, sigma):
    """计算任意时刻t的标准差"""
    t = torch.tensor(t, device=device)
    return torch.sqrt((sigma ** (2 * t) - 1.) / 2. / np.log(sigma))


def diffusion_coeff(t, sigma):
    """计算任意时刻t的扩散系数"""
    return torch.tensor(sigma ** t, device=device)


sigma = 25.0
marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)  # 构建无参函数
diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma)  # 构建无参函数


def loss_fn(score_model, x, marginal_prob_std_fn, eps=1e-5):
    """计算损失函数"""

    # 1. 从[0,1]中随机采样batch_size个数
    random_t = torch.rand(x.shape[0], device=device) * (1. - eps) + eps  # 生成随机数且不为0或1
    # 2. 基于重参数技巧采样分布p_t(x)的随机样本perturbed_x
    z = torch.randn_like(x, device=device)  # 生成正态分布的随机数
    std = marginal_prob_std_fn(random_t)  # 计算标准差
    perturbed_x = x + z * std[:, None, None, None]  # 计算扰动后x
    # 3. 将当前的加噪样本和时间输入到score_model中，得到分数
    score = score_model(perturbed_x, random_t)
    # 4. 计算损失函数
    loss = torch.mean(torch.sum((score * std[:, None, None, None] + z) ** 2, dim=[1, 2, 3]))  # 计算损失函数

    return loss


class EMA(nn.Module):
    def __init__(self, model, decay=0.9999, device=None):
        super(EMA, self).__init__()
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device
        if self.device is not None:
            self.module.to(self.device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)


# 基于随机微分方程来生成数据（Euler方法）
num_steps = 1000


def euler_sampler(score_model, marginal_prob_std, diffusion_coeff, device, batch_size=64, num_steps=1000, eps=1e-3):
    """Euler方法采样"""
    # 1. 定义初试时间1和先验分布的随机样本
    t = torch.ones(batch_size, device=device)
    init_x = torch.randn(batch_size, 1, 28, 28, device=device) \
             * marginal_prob_std(t)[:, None, None, None]

    # 2. 定义采样的逆时间网格以及每一步的时间步长
    time_steps = torch.linspace(1., eps, num_steps, device=device)
    step_size = time_steps[0] - time_steps[1]

    # 3. 根据Euler方法迭代采样
    x = init_x
    with torch.no_grad():
        for time_step in tqdm.tqdm(time_steps):
            batch_time_step = torch.ones(batch_size, device=device) * time_step
            g = diffusion_coeff(batch_time_step)  # 计算扰动系数
            mean_x = x + (g ** 2)[:, None, None, None] * score_model(x, batch_time_step) * step_size  # 计算均值
            x = mean_x + torch.sqrt(step_size) * g[:, None, None, None] * torch.randn_like(x, device=device)  # 计算扰动后的x

    # 4. 最后一步的期望作为输出
    return mean_x


# 融合Euler方法和郎之万方法（predictor-corrector方法）
signal_to_noise_ratio = 0.16
num_steps = 1000


def pc_sampler(score_model, marginal_prob_std, diffusion_coeff, device, z=None,
               batch_size=64, num_steps=1000, num_corrector_steps=10, snr=0.16, eps=1e-3):
    """predictor-corrector方法采样"""
    # 1. 定义初试时间1和先验分布的随机样本
    t = torch.ones(batch_size, device=device)
    if z is None:
        init_x = torch.randn(batch_size, 1, 28, 28, device=device) \
                * marginal_prob_std(t)[:, None, None, None]
    else:
        init_x = (z * marginal_prob_std(t)[:, None, None, None]).to(device)

    # 2. 定义采样的逆时间网格以及每一步的时间步长
    time_steps = torch.linspace(1., eps, num_steps, device=device)
    step_size = time_steps[0] - time_steps[1]

    # 3. 根据predictor-corrector方法迭代采样, 重复交替郎之万方法和Euler方法
    x = init_x
    with torch.no_grad():
        for time_step in tqdm.tqdm(time_steps):
            batch_time_step = torch.ones(batch_size, device=device) * time_step
            # Corrector step(Langevin MCMC)
            grad = score_model(x, batch_time_step)
            grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
            noise_norm = np.sqrt(np.prod(x.shape[1:]))
            langevin_step_size = 2 * (snr * noise_norm / grad_norm) ** 2  # 保证信噪比

            for _ in range(num_corrector_steps):
                x = x + langevin_step_size * grad + torch.sqrt(2 * langevin_step_size) * torch.randn_like(x,
                                                                                                          device=device)
                grad = score_model(x, batch_time_step)
                grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
                noise_norm = np.sqrt(np.prod(x.shape[1:]))
                langevin_step_size = 2 * (snr * noise_norm / grad_norm) ** 2

            # Predictor step(Euler Maruyama)
            g = diffusion_coeff(batch_time_step)
            x_mean = x + (g ** 2)[:, None, None, None] * score_model(x, batch_time_step) * step_size
            x = x_mean + torch.sqrt(g ** 2 * step_size)[:, None, None, None] * torch.randn_like(x, device=device)

    # 4. 最后一步的期望作为输出
    return x_mean


# 基于伴随常微分数值解法来生成数据

error_tolerance = 1e-5


def ode_sampler(score_model, marginal_prob_std, diffusion_coeff, device, batch_size=64, atol=error_tolerance,
                rtol=error_tolerance, z=None, eps=1e-3):
    """基于伴随常微分数值解法来生成数据"""
    # 1. 定义初试时间1和先验分布的随机样本
    t = torch.ones(batch_size, device=device)
    if z is None:
        init_x = torch.randn(batch_size, 1, 28, 28, device=device) \
                 * marginal_prob_std(t)[:, None, None, None]
    else:
        init_x = z
    shape = init_x.shape

    # 2. 定义分数预测和常微分函数
    def score_eval_wrapper(sample, time_steps):
        sample = torch.tensor(sample, device=device, dtype=torch.float32).reshape(shape)
        time_step = torch.tensor(time_steps, device=device, dtype=torch.float32).reshape((sample.shape[0],))
        with torch.no_grad():
            score = score_model(sample, time_step)
        return score.cpu().numpy().reshape((-1,)).astype(np.float64)

    def ode_func(t, x):
        time_steps = np.ones((shape[0],)) * t
        g = diffusion_coeff(torch.tensor(t)).cpu().numpy()
        return score_eval_wrapper(x, time_steps) * (g ** 2) * (-0.5)

    # 3. 调用常微分算子解出t=eps时的样本
    res = integrate.solve_ivp(ode_func, (1., eps), init_x.reshape(-1).cpu().numpy(), rtol=rtol, atol=atol,
                              method='RK45')
    print(f'Number of function evaluations: {res.nfev}')

    x = torch.tensor(res.y[:, -1], device=device).reshape(shape)
    return x


def sample(sampler, sample_batch_size, score_model, marginal_prob_std_fn, diffusion_coeff_fn, device):
    """采样函数"""
    if sampler == 'euler':
        sampler = euler_sampler
        name = 'Euler'
    elif sampler == 'pc':
        sampler = pc_sampler
        name = 'PC'
    elif sampler == 'ode':
        sampler = ode_sampler
        name = 'ODE'
    else:
        raise ValueError('Unknown sampler')
    score_model.eval()
    with torch.no_grad():
        t1 = time.time()
        samples = sampler(score_model, marginal_prob_std_fn, diffusion_coeff_fn, device)
        t2 = time.time()
        print(f'Elapsed time: {t2 - t1:.2f}s')

        samples = samples.clamp(0., 1.)
        sample_grid = make_grid(samples, nrow=int(np.sqrt(sample_batch_size)))

        plt.figure(figsize=(6, 6))
        plt.axis('off')
        plt.imshow(sample_grid.cpu().numpy().transpose((1, 2, 0)), vmin=0., vmax=1.)
        plt.savefig('output/' + name + '.png')


if __name__ == '__main__':

    # 在MNIST数据集上训练
    score_model = torch.nn.DataParallel(ScoreNet(marginal_prob_std=marginal_prob_std_fn))
    score_model.to(device)
    if os.path.exists('ckpt.pth'):
        score_model.load_state_dict(torch.load('ckpt.pth'))
        print('Load model from ckpt.pth')

    n_epochs = 30
    batch_size = 512
    lr = 1e-4

    dataset = datasets.MNIST(root='data', train=True, download=True, transform=transforms.ToTensor())
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    optimizer = torch.optim.Adam(score_model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=n_epochs*len(data_loader), eta_min=1e-6)

    ema = EMA(score_model, device=device)
    score_model.train()
    for epoch in range(n_epochs):

        t = tqdm.tqdm(data_loader)
        for x, _ in t:
            x = x.to(device)
            t.set_description(f'Epoch {epoch + 1}/{n_epochs}')
            loss = loss_fn(score_model, x, marginal_prob_std_fn)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.update(score_model)
            t.set_postfix(loss=loss.item())

        scheduler.step()

        torch.save(score_model.state_dict(), 'ckpt.pth')

    # 导入模型并用于生成数据
    ckpt = torch.load('ckpt.pth', map_location=device)
    score_model.load_state_dict(ckpt)
    score_model.eval()

    sample_batch_size = 64
    sample('euler', sample_batch_size, score_model, marginal_prob_std_fn, diffusion_coeff_fn, device)
    sample('pc', sample_batch_size, score_model, marginal_prob_std_fn, diffusion_coeff_fn, device)
    sample('ode', sample_batch_size, score_model, marginal_prob_std_fn, diffusion_coeff_fn, device)

    # 原图和生成图对比
    num_pic = 10
    original = next(iter(data_loader))[0].to(device)[0:num_pic]
    t = torch.ones(original.shape[0], device=device)*0.5
    z = torch.randn_like(original, device=device) * marginal_prob_std_fn(t)[:, None, None, None]
    sampled_x = pc_sampler(score_model, marginal_prob_std_fn, diffusion_coeff_fn, device, z=z, batch_size=10)  # 用PC算法采样
    sampled_x = sampled_x.clamp(0., 1.)
    pic_list = []
    for i in range(num_pic):
        pic_list.append(original[i])
        pic_list.append(z[i])
        pic_list.append(sampled_x[i])
    sample_grid = make_grid(pic_list, nrow=3)
    plt.figure(figsize=(6, 6))
    plt.axis('off')
    plt.imshow(sample_grid.cpu().numpy().transpose((1, 2, 0)), vmin=0., vmax=1.)
    plt.savefig('sample.png')




