# facenet
# 1. 输入一张人脸图片
# 2. 通过深度卷积网络（主干特征提取网络）提取特征
# 3. L2范数归一化
# 4. 得到一个128维的向量

# 可以使用分类器辅助训练，将128维的向量进行不同人脸的分类

# 这里的代码主要用于人脸区分，不用于人脸识别，真实的人脸识别还需要先检测人脸位置，然后再进行人脸识别

# 使用train之前先用txt_annotation.py生成对应的txt文件，然后再使用train.py进行训练
# 训练记录和模型保存在logs下
# 训练数据集在datasets下，一个文件夹一个人脸
# 在lfw文件夹中为lfw数据集，使用lfw数据集，进行训练中或训练后测试，lfw_pair.txt为lfw数据集的配对文件，在model_data文件夹中

# 可以使用eval_LFW.py进行测试，roc曲线存在model_data文件夹中

# 使用predict.py计算两张图片的相似度
# img中是一些图片，可以用其中的图片进行predict.py的测试
# facenet.py中的类用于使用backbone进行人脸区分，在predict.py中使用

# nets文件夹下有facenet的backbone网络(mobilenet和inception_resnet)，model_data中有预训练的模型
# 其中facenet.py用于使用backbone前向传播，得到128维的向量，用于训练
# facenet_training.py中有triplet_loss函数，网络参数初始化函数，训练记录保存函数

# utils文件夹中的dataloader有FacenetDataset和LFWDataset，前者返回3倍batch_size的图片和标签，后者返回两种图片和是否为同一个人
# utils_fit是训练函数，包括训练和验证
# 使用utils_metrics.py中的evaluate可以得到准确率，召回率，最佳阈值等

# summary.py可以查看模型的参数
