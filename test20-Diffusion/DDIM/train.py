import torchvision
import numpy as np
from DDIM_model import Model, DiffusionProcessDDIM
from utils import AverageMeter, ProgressMeter, make_grid, imshow
import torch
import random
import os
import matplotlib.pyplot as plt

seed = 999
random.seed(seed)
torch.manual_seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device('cuda:0')


def main():
    beta_1 = 1e-4
    beta_T = 0.02
    T = 1000
    shape = (1, 16, 16)
    eta = 1  # sigma
    tau = 1  # tau = 1, 5, 10, 20, 50, 100 加速倍率
    model = Model(device, beta_1, beta_T, T)
    process = DiffusionProcessDDIM(beta_1, beta_T, T, model, device, shape, eta, tau)
    optim = torch.optim.Adam(model.parameters(), lr=0.0001)

    total_iteration = 5000
    current_iteration = 0
    display_iteration = 1000

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((shape[-2], shape[-1])),
        torchvision.transforms.ToTensor()
    ])

    sampling_number = 16
    only_final = True

    dataset = torchvision.datasets.MNIST(root='./MNIST', train=True, download=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, drop_last=True, num_workers=3)
    dataiterator = iter(dataloader)

    losses = AverageMeter('Loss', ':.4f')
    progress = ProgressMeter(total_iteration, [losses], prefix='Iteration ')

    # Training
    while current_iteration != total_iteration:
        try:
            data = next(dataiterator)
        except:
            dataiterator = iter(dataloader)
            data = next(dataiterator)
        data = data[0].to(device=device)
        loss = model.loss_fn(data)

        optim.zero_grad()
        loss.backward()
        optim.step()

        losses.update(loss.item())
        progress.display(current_iteration)
        current_iteration += 1

        if current_iteration % display_iteration == 0:
            process = DiffusionProcessDDIM(beta_1, beta_T, T, model, device, shape, eta)
            sample = process.sampling(sampling_number, only_final=only_final)
            imshow(sample, sampling_number)
            losses.reset()

            torch.save(model.state_dict(), 'model.pth')

    # Generate samples (DDIM)
    sampling_number = 64
    z_first = torch.randn(sampling_number, 1, 16, 16).to(device=device)
    eta = 0
    processDDIM = DiffusionProcessDDIM(beta_1, beta_T, T, model, device, shape, eta)
    x_DDIM = processDDIM.sampling(sampling_number, sample=z_first, only_final=only_final)
    imshow(x_DDIM, sampling_number, save=True)

    # Accelarate sampling
    eta = 0
    tau = 5  # tau = 1, 5, 10, 20, 50, 100 加速倍率
    processDDIM = DiffusionProcessDDIM(beta_1, beta_T, T, model, device, shape, eta, tau=tau, scheduling='exp')
    x_DDIM_5X = processDDIM.sampling(sampling_number, sample=z_first, only_final=only_final)
    imshow(x_DDIM_5X, sampling_number, save=True, name='DDIM_5X')

    # Original -> Encoding -> Reconstruction
    tau = 1
    processDDIM = DiffusionProcessDDIM(beta_1, beta_T, T, model, device, shape, eta, tau=tau)
    x_original = x_DDIM.clamp(0, 1)

    x_original2fw = processDDIM.probabilityflow(x_original, reverse=False)  # Forward
    x_ofiginal2fw2bw = processDDIM.probabilityflow(x_original2fw, reverse=True)  # Backward

    x_original_grid = make_grid(x_original)
    x_original2fw_grid = make_grid(x_original2fw)
    x_ofiginal2fw2bw_grid = make_grid(x_ofiginal2fw2bw)
    fig, axs = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

    axs[0].imshow(x_original_grid, cmap='gray')
    axs[1].imshow(x_original2fw_grid, cmap='gray')
    axs[2].imshow(x_ofiginal2fw2bw_grid, cmap='gray')

    for i in range(3):
        axs[i].axis(False)

    axs[0].set_title('original', fontsize=15)
    axs[1].set_title('original -> forward', fontsize=15)
    axs[2].set_title('original -> forward -> backward', fontsize=15)

    plt.savefig('Probability Flow ODE-o2f2b.png')
    plt.show()


if __name__ == '__main__':
    main()
