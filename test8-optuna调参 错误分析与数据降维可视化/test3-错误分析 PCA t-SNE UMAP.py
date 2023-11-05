import torch
import torchvision
import torch.nn as nn
from visdom import Visdom
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP
import matplotlib.pyplot as plt


class Net(nn.Module):  # 网络结构
    def __init__(self, conv1_channels, conv2_channels, conv3_channels, conv4_channels, FC1_channels, FC2_channels,
                 dropout_rate1, dropout_rate2, activation):
        super(Net, self).__init__()
        if activation == 'ReLU':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'LeakyReLU':
            self.activation = nn.LeakyReLU(inplace=True)
        elif activation == 'SELU':
            self.activation = nn.SELU(inplace=True)
        self.model = nn.Sequential(
            nn.Conv2d(1, conv1_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(conv1_channels),
            nn.MaxPool2d(2),
            self.activation,

            nn.Conv2d(conv1_channels, conv2_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(conv2_channels),
            nn.MaxPool2d(2),
            self.activation,

            nn.Conv2d(conv2_channels, conv3_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(conv3_channels),
            nn.MaxPool2d(2),
            self.activation,

            nn.Conv2d(conv3_channels, conv4_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(conv4_channels),
            nn.MaxPool2d(2),
            self.activation,

            nn.Flatten(),

            nn.Linear(conv4_channels, FC1_channels),
            self.activation,
            nn.Dropout(dropout_rate1),

            nn.Linear(FC1_channels, FC2_channels),
            self.activation,
            nn.Dropout(dropout_rate2),

            nn.Linear(FC2_channels, 10)

        )

    def forward(self, x):
        return self.model(x)


def get_confuse_matrix(model, test_loader):
    confusion_matrix = torch.zeros(10, 10)
    with torch.no_grad():
        for data, target in test_loader:
            outputs = model(data)
            pred = outputs.argmax(dim=1)
            for t, p in zip(target.view(-1), pred.view(-1)):
                confusion_matrix[t.int(), p.int()] += 1
    return confusion_matrix.int()


def denormalize(x, mean=0.1307, std=0.3081):
    return x * std + mean


def visualize_error(model, test_loader, viz):
    model.eval()
    with torch.no_grad():
        error_data = []
        label_list = []
        for data, target in test_loader:
            outputs = model(data)
            pred = outputs.argmax(dim=1)
            for i in range(len(target)):
                if pred[i] != target[i]:
                    error_data.append(denormalize(data[i]))
                    label_list.append('pred: {} / label: {}'.format(pred[i].item(), target[i].item()))

        error_data = torch.stack(error_data, dim=0)
        viz.images(error_data, nrow=4, opts=dict(title='error_data'), win='error_data')
        viz.text(str(label_list), opts=dict(title='error_label'), win='error_label')


def get_data_pred(model, test_loader):
    model.eval()
    with torch.no_grad():
        data = []
        target = []
        pred_list = []
        for x, y in test_loader:
            data.append(x)
            target.append(y)
            outputs = model(x)
            pred = outputs.argmax(dim=1)
            pred_list.append(pred)
        data = torch.cat(data, dim=0)
        target = torch.cat(target, dim=0)
        pred_list = torch.cat(pred_list, dim=0)
        data = data.view(data.size(0), -1)
        data = data.numpy()
        target = target.numpy()
        pred_list = pred_list.numpy()
        print(f'data.shape: {data.shape}, target.shape: {target.shape}, pred_list.shape: {pred_list.shape}')
        # data.shape: (10000, 784), target.shape: (10000,), pred_list.shape: (10000,)

    return data, target, pred_list


def pca_evaluation(data, target, pred_list):
    pca = PCA(n_components=3)  # 降维到3维,用2维也可以
    pca.fit(data)
    data = pca.transform(data)
    print(f'data.shape: {data.shape}')  # data.shape: (10000, 3)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=target, cmap='rainbow')
    ax.set_title('target_PCA')
    ax.axes.xaxis.set_ticks([])# 去掉x坐标轴刻度
    ax.axes.yaxis.set_ticks([])# 去掉y坐标轴刻度
    ax.axes.zaxis.set_ticks([])# 去掉z坐标轴刻度
    fig.colorbar(ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=target, cmap='rainbow'))# 添加颜色标签
    plt.savefig('target_PCA.png')

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=pred_list, cmap='rainbow')
    ax.set_title('pred_PCA')
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])
    ax.axes.zaxis.set_ticks([])
    fig.colorbar(ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=pred_list, cmap='rainbow'))
    plt.savefig('pred_PCA.png')


def t_SNE_evaluation(data, target, pred_list):
    tsne = TSNE(n_components=2, init='pca', learning_rate='auto', perplexity=30.0)
    # 降维到2维,初始化方式为pca,学习率为自动调整，
    # perplexity:困惑度，在t-SNE中，perplexity的定义是“number of nearest neighbors”，也就是说每个元素考虑多少个邻居
    # 也可理解为一个cluster中有多少个元素，但是相同数量和维数的数据(即使都应该分为10类)，分布不同，最好的perplexity也不同，一般在5-50之间
    # 其实perplexity可以理解为是一个用来平衡 t-SNE 关注局部变换还是关注全局变换的权重, 越小越关注局部变换,越大越关注全局变换
    # 也可以理解为，perplexity 是刻画每一个点的邻接点的个数的参数。它对 t-SNE 的结果的影响很细致，
    data = tsne.fit_transform(data)
    print(f'data.shape: {data.shape}')  # data.shape: (10000, 2)

    plt.figure()
    plt.scatter(data[:, 0], data[:, 1], c=target, cmap='rainbow')# 画散点图,颜色为target,target为0-9,故颜色为0-9,target[i]对应(data[i,0],data[i,1])的坐标
    plt.title('target_t-SNE')
    plt.colorbar()# 显示颜色条
    plt.xticks([])# 不显示x轴刻度
    plt.yticks([])# 不显示y轴刻度
    plt.savefig('target_t-SNE.png')

    plt.figure()
    plt.scatter(data[:, 0], data[:, 1], c=pred_list, cmap='rainbow')
    plt.title('pred_t-SNE')
    plt.colorbar()# 显示颜色条
    plt.xticks([])# 不显示x轴刻度
    plt.yticks([])# 不显示y轴刻度
    plt.savefig('pred_t-SNE.png')


def umap_evaluation(data, target, pred_list):
    reducer = UMAP(n_components=2, n_neighbors=5, min_dist=0.1, metric='euclidean')
    # 降维到2维,n_neighbors是高维空间中每个点的邻居数目,越大保留的全局信息越多,越小保留的局部信息越多,太小会导致虚假聚类
    # min_dist是低维空间中点的最小距离,越大越稀疏,越小越紧凑,越小相似的点越聚集,越大相似的点越稀疏
    # metric是距离度量方式，可选参数有euclidean,manhattan,chebyshev,minkowski,canberra,braycurtis,correlation,cosine,dice等
    data = reducer.fit_transform(data)
    print(f'data.shape: {data.shape}')  # data.shape: (10000, 2)

    plt.figure()
    plt.scatter(data[:, 0], data[:, 1], c=target, cmap='rainbow')
    plt.title('target_umap')
    plt.colorbar()# 显示颜色条
    plt.xticks([])# 不显示x轴刻度
    plt.yticks([])# 不显示y轴刻度
    plt.savefig('target_umap.png')

    plt.figure()
    plt.scatter(data[:, 0], data[:, 1], c=pred_list, cmap='rainbow')
    plt.title('pred_umap')
    plt.colorbar()# 显示颜色条
    plt.xticks([])# 不显示x轴刻度
    plt.yticks([])# 不显示y轴刻度
    plt.savefig('pred_umap.png')


def main():
    # best_params:
    # {'FC1_channels': 209, 'FC2_channels': 100, 'activation': 'ReLU', 'conv1_channels': 39, 'conv2_channels': 121,
    # 'conv3_channels': 245, 'conv4_channels': 418, 'dropout_rate1': 0.23533337796604623, 'dropout_rate2': 0.4791637774135236,
    # 'lr': 0.0001589867782192497, 'optimizer': 'AdamW', 'scheduler': 'CosineAnnealingWarmRestarts',
    # 'weight_decay': 7.333130236322882e-05}
    model = Net(39, 121, 245, 418, 209, 100, 0.23533337796604623, 0.4791637774135236, 'ReLU')
    model.load_state_dict(torch.load('models/model No.9 0.994.pth'))
    test_db = torchvision.datasets.MNIST('mnist_data/', train=False,
                                         download=True,  # 下载数据集
                                         transform=torchvision.transforms.Compose([
                                             torchvision.transforms.ToTensor(),
                                             torchvision.transforms.Normalize(
                                                 (0.1307,), (0.3081,))
                                         ]))
    test_loader = torch.utils.data.DataLoader(test_db, batch_size=128, shuffle=True)

    confuse_matrix = get_confuse_matrix(model, test_loader)# 生成混淆矩阵
    print(confuse_matrix)
    viz = Visdom()
    # 生成热力图
    viz.heatmap(confuse_matrix, opts=dict(title='confuse_matrix', columnnames=list(range(10)), rownames=list(range(10)), showlegend=True))

    visualize_error(model, test_loader, viz)

    data, target, pred_list = get_data_pred(model, test_loader)

    # 数据降维，这里为了可视化，降维到2、3维，实际应用的时候不会降到这么低维

    # 使用PCA降维并可视化
    pca_evaluation(data, target, pred_list)

    # 使用t-SNE聚类降维并可视化
    t_SNE_evaluation(data, target, pred_list)

    # 使用umap聚类降维并可视化
    umap_evaluation(data, target, pred_list)


if __name__ == '__main__':
    main()
