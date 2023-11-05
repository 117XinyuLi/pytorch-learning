import os
from torch.utils.data import DataLoader
from dataset import *
from yolo_v3_net import *
from config import *
from torch.utils.tensorboard import SummaryWriter

save_path = 'params/yolo_v3_net.pth'


def YoloLoss(output, target, device, c=C):
    # output
    # [batch_size, 3*(5+num_classes), 13, 13] 大目标
    # or [batch_size, 3*(5+num_classes), 26, 26] 中目标
    # or [batch_size, 3*(5+num_classes), 52, 52] 小目标
    # target
    # [batch_size, 13, 13, 3, 5+num_classes] 大目标
    # or [batch_size, 26, 26, 3, 5+num_classes] 中目标
    # or [batch_size, 52, 52, 3, 5+num_classes] 小目标
    output = output.permute(0, 2, 3, 1).contiguous().view(output.size(0), output.size(2), output.size(3), 3, -1)
    # output: [batch_size, 13, 13, 3, 5+num_classes]
    mask_obj = target[..., 0] > 0
    #

    # 1.是否包含目标的损失
    loss_p_fun = nn.BCELoss().to(device)
    loss_p = loss_p_fun(torch.sigmoid(output[..., 0]), target[..., 0])
    # 2.目标中心点和长宽的损失，只计算包含目标的格子，不关心无目标的格子中的内容
    loss_box_fun = nn.MSELoss().to(device)
    loss_box = loss_box_fun(output[..., 1:5][mask_obj], target[..., 1:5][mask_obj])
    # 3.多分类的损失，只计算包含目标的格子，不关心无目标的格子中的内容
    loss_c_fun = nn.CrossEntropyLoss().to(device)
    loss_c = loss_c_fun(output[..., 5:][mask_obj], torch.argmax(target[..., 5:][mask_obj], dim=-1, keepdim=True).squeeze(-1))

    return c*loss_p + (1-c)*(loss_box + loss_c)*0.5


if __name__ == '__main__':
    summary_writer = SummaryWriter(log_dir='logs')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = YoloDataset()
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

    net = YoloV3Net(CLASS_NUM).to(device)
    if os.path.exists(save_path):
        net.load_state_dict(torch.load(save_path))
        print('load params success')

    epochs = 100
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=epochs*len(dataloader), T_mult=1, eta_min=1e-5)

    global_step = 0
    for epoch in range(epochs):
        loss = 0
        net.train()
        for target_13, target_26, target_52, img in dataloader:
            img = img.to(device)
            target_13 = target_13.to(device)
            target_26 = target_26.to(device)
            target_52 = target_52.to(device)

            output_13, output_26, output_52 = net(img)
            loss_13 = YoloLoss(output_13.float(), target_13.float(), device)
            loss_26 = YoloLoss(output_26.float(), target_26.float(), device)
            loss_52 = YoloLoss(output_52.float(), target_52.float(), device)
            loss = loss_13 + loss_26 + loss_52

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            summary_writer.add_scalar('loss', loss.item(), global_step)
            # 在终端输入 tensorboard --logdir='logs的绝对路径'以查看，tensorboard依赖于tensorflow
            global_step += 1

        print(f'epoch: {epoch}, loss: {loss.item()}')
        torch.save(net.state_dict(), save_path)

