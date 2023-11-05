import torch
from torch import nn, optim, autograd
import numpy as np
import random
import visdom
from matplotlib import pyplot as plt

h_dim = 400
batchsz = 512
viz = visdom.Visdom()


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            # z:[b, 2] => [b, 2] # 第一个2是输入的维度，第二个2是输出为了方便后面的可视化
            nn.Linear(2, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, 2)
        )

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            # x:[b, 2] => [b, 1]
            nn.Linear(2, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


# 黄色点是高斯分布的均值点，绿色加号是生成的点，应比较靠近黄色点
def generate_image(D, G, xr, epoch):
    """
    Generates and saves a plot of the true distribution, the generator, and the
    critic.
    """
    N_POINTS = 128
    RANGE = 3
    plt.clf()

    points = np.zeros((N_POINTS, N_POINTS, 2), dtype='float32')
    points[:, :, 0] = np.linspace(-RANGE, RANGE, N_POINTS)[:, None]
    points[:, :, 1] = np.linspace(-RANGE, RANGE, N_POINTS)[None, :]
    points = points.reshape((-1, 2))
    # (16384, 2)
    # print('p:', points.shape)

    # draw contour
    with torch.no_grad():
        points = torch.Tensor(points).cuda()  # [16384, 2]
        disc_map = D(points).cpu().numpy()  # [16384]
    x = y = np.linspace(-RANGE, RANGE, N_POINTS)
    cs = plt.contour(x, y, disc_map.reshape((len(x), len(y))).transpose())
    plt.clabel(cs, inline=1, fontsize=10)
    # plt.colorbar()

    # draw samples
    with torch.no_grad():
        z = torch.randn(batchsz, 2).cuda()  # [b, 2]
        samples = G(z).cpu().numpy()  # [b, 2]
    plt.scatter(xr[:, 0], xr[:, 1], c='orange', marker='.')
    plt.scatter(samples[:, 0], samples[:, 1], c='green', marker='+')

    viz.matplot(plt, win='contour', opts=dict(title='p(x):%d' % epoch))


def data_generator():
    """
    8-Gaussian mixture model
    :return:point (512,2)
    """
    scale = 2.
    centers = [
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1. / np.sqrt(2), 1. / np.sqrt(2)),
        (1. / np.sqrt(2), -1. / np.sqrt(2)),
        (-1. / np.sqrt(2), 1. / np.sqrt(2)),
        (-1. / np.sqrt(2), -1. / np.sqrt(2))
    ]
    centers = [(scale * x, scale * y) for x, y in centers] # 8个中心点进行缩放x为横坐标，y为纵坐标

    while True:
        dataset = []
        for i in range(batchsz):
            point = np.random.randn(2) * .02
            center = random.choice(centers)
            point[0] += center[0]
            point[1] += center[1]
            dataset.append(point)
        dataset = np.array(dataset).astype('float32')
        dataset /= 1.414
        yield dataset# 虽然为死循环，但是会从yield处返回数据，下一次调用时从yield处继续执行


def gradient_penalty(D, xr, xf):
    """
    :param D:
    :param xr:[b, 2]
    :param xf:[b, 2]
    :return: gp
    """
    t = torch.rand(batchsz, 1).cuda()# 生成batchsz个0-1之间的随机数,用于插值
    t = t.expand_as(xr)# 扩展为与xr相同的形状
    mid = t * xr + (1 - t) * xf# 插值
    mid.requires_grad_()# 设置为可求导

    pred = D(mid)# [b, 1]
    grads = torch.autograd.grad(
        outputs=pred, inputs=mid,
        grad_outputs=torch.ones_like(pred), create_graph=True, retain_graph=True, only_inputs=True)[0]# 求导 [b, 2]
    # 一定要设置retain_graph=True，否则第二次backward会报错
    # 一定要设置grad_outputs=torch.ones_like(pred)，否则只能对标量求导
    # create_graph=True，是为了在求导的时候创建新的计算图，这样可以继续求导
    # only_inputs=True，只对mid求导，不对pred求导

    gp = ((grads.norm(2, dim=1) - 1) ** 2).mean()# 求梯度的范数，然后求平均值

    return gp


def main():
    torch.manual_seed(23)
    np.random.seed(23)
    data_iter = data_generator()
    x = next(data_iter)
    print(x.shape)# (512, 2)

    G = Generator().cuda()
    D = Discriminator().cuda()
    optim_G = optim.Adam(G.parameters(), lr=5e-4, betas=(0.5, 0.9))# betas是adam的参数, GAN中一般这么设置
    optim_D = optim.Adam(D.parameters(), lr=5e-4, betas=(0.5, 0.9))

    viz.line([[0, 0]], [0], win='loss', opts=dict(title='loss', legend=['D', 'G']))
    xr = 0

    for epoch in range(30000):
        # train D firstly
        loss_D = 0
        for _ in range(5):# 训练5次D
            # 1.1 train D with real data
            x = next(data_iter)
            xr = torch.from_numpy(x).cuda()
            # x:[b, 2]=>[b, 1]
            predr = D(xr)
            # max predr
            lossr = -predr.mean()# 真实数据的loss, 加符号是因为是要最大化, SGD默认是最小化, 为了方便, 防止梯度爆炸, 没有加log

            # 1.2 train D with fake data
            z = torch.randn(batchsz, 2).cuda()
            xf = G(z)
            # xf:[b, 2]=>[b, 1]
            predf = D(xf.detach())# detach()是为了不让G更新
            # min predf
            lossf = predf.mean()# 生成数据的loss

            # 1.3 gradient penalty
            gp = gradient_penalty(D, xr, xf.detach())

            # aggregate loss
            loss_D = lossr + lossf + 0.2 * gp

            # optimize
            optim_D.zero_grad()
            loss_D.backward()
            optim_D.step()

        # train G
        z = torch.randn(batchsz, 2).cuda()
        xf = G(z)
        predf = D(xf)# 这里不用detach()是因为G要更新，而D不用更新，我们train G的时候只更新G的参数就行，更新D的参数之前先清零梯度就行了
        # max predf
        loss_G = -predf.mean()# 负号是因为要最大化

        # optimize
        optim_G.zero_grad()
        loss_G.backward()
        optim_G.step()

        if epoch % 100 == 0:
            viz.line([[loss_D.item(), loss_G.item()]], [epoch], win='loss', update='append')
            print(epoch, 'loss_D:', loss_D.item(), 'loss_G:', loss_G.item())
            generate_image(D, G, xr.cpu().numpy(), epoch)


if __name__ == '__main__':
    main()
