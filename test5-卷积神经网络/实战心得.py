"""
关于训练：
1. 可以先在模型文件中测试能不能跑通，然后再训练
2. batch size可以随着epoch增大，但是batch size不能太大，否则会导致显存不够，训练效果也不一定好，一般batch size加倍，学习率也需要加倍
   显存不足的时候可以按test8中的方法，累计梯度，近似实现batch size增大，或者checkpoint，
   但是会导致训练速度变慢，有BN层的话，会导致BN层的参数不准确，dropout每次关闭的神经元不一样等
3. 一开始可以先不dropout，小weight decay，先解决bias的问题，训一个模型(dev acc不变时就可停下了，loss不要太低)，
   然后再加入dropout(一般对于FC)和weight decay，看看效果，解决variance的问题(dev acc不变时就可停下了，loss不要太低)，
   若再出现high bias，减小weight decay，减小dropout，不断训练，解决bias的问题(直到dev acc不变，loss不要太低)，也可以增加模型的复杂度，或者增加训练的epoch，
   high variance的话，可以增加数据，一般来说一个良好正则化的神经网络，模型越深越好，事实上，可以找模型最适合的深度
   重复上述过程，直到loss比较低，加入dropout和weight decay，长时间训练(真的很长)后停止
4. model.train()和model.eval()的要随时切换
5. 随时保存模型，以防万一
6. RNN中每个batch执行后，若模型返回值有hidden，需要hidden=hidden.detach()，隐藏状态不用计算梯度

关于模型与数据集：
1. 一般来说，分为train dev test三个集合，train用来训练，dev用来调参，test用来测试，test集合不能参与训练和调参
2. DataLoader中num_workers可以指定多线程读取数据
3. 一个良好正则化的神经网络，模型越深越好，出现high bias/high variance的情况，可以增加模型的复杂度，或者增加训练的epoch，事实上，可以找模型最适合的深度
4. Conv2d紧跟着BN层时，可以使用nn.Conv2d(...,bias=False),这样可以减少参数量，加快训练速度，防止对bias过度依赖
5. 随时del不用的变量，以防内存不够
6. 使用迁移学习(数据量不够的时候可以使用)，最好输入的图片大小和原模型一样，否则相当于需要重新训练
7. 可使用timm库，里面有很多模型，可以直接使用，也可以使用预训练模型，也可以使用自己的模型
8. 可使用torchsummary库，可以打印模型的参数量
9. 可使用vit_pytorch库，可以使用Vision Transformer模型

其他tricks：
1. 学习率递减、数据增强、正则化、dropout、weight decay、batch size、epoch数、模型参数、随机种子等都是可以调的
2. error analysis，分析错误的样本，看看是哪些样本出错了，然后再对这些样本进行改进
3. 余弦退火(学习率可以一直递减(一般是这样的)，或者增减不断变化)、
   onecycle(在训练的总epoch中，lr先增大到原来的10倍，然后减小到小于初始值几个数量级，小心一开始的lr不要太大eg.用3e-4，一开始loss可能会上升，后面会下降)，
   迁移学习时，谨慎使用(需要好好调lr)，一般学习率衰减足矣
4. AdamW使用的是weight decay(权重衰减)，而不是L2正则化，使用weight decay的AdamW在训练速度经常比Adam要好,
   Adam和AdamW的weight decay对解决high variance的问题有帮助，但是帮助有限
   Adam的学习率初始经常为3e-4，SGD的学习率初始经常为1e-2
5. 混合精度训练，可以加快训练速度，但是可能会导致精度下降，可以尝试一下
   scaler = torch.cuda.amp.GradScaler()
   for data, target in train_loader:
       optimizer.zero_grad()
       with torch.cuda.amp.autocast():
            output = model(data)
            loss = criterion(output, target)
       scaler.scale(loss).backward()
       scaler.step(optimizer)
       scaler.update()
6. 输入大小不变时，可以torch.backends.cudnn.benchmark = True，可以让系统找最快的计算卷积的方式，加快训练速度(前期训练速度会变慢，后期训练速度会变快)
7. 梯度裁剪，可以防止RNN梯度爆炸，可以尝试一下
    for p in model.parameters():
        torch.nn.utils.clip_grad_norm_(p, max_norm=1.0)# 将梯度的模长限制在1.0以内
8. 清空缓存 torch.cuda.empty_cache()
9. 使用多GPU，先os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"，确定使用GPU数量和类型，然后model = nn.DataParallel(model)或者nn.DistributedDataParallel(model)，
   然后对于model，model=model.cuda()，对于数据，data=data.cuda()，对于target，target=target.cuda()

"""