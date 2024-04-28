import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision.transforms import transforms
from tqdm import tqdm

import SRResNet
import main
from main import PreprocessDataset

path = '/home/featurize/pythonProject5/Urban100/image_SRF_2/'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH = 128
#构建数据集
processDataset = main.PreprocessDataset(path)
trainData = DataLoader(processDataset,batch_size=BATCH)
net = SRResNet.SRResNet();
net = net.to(device)  # 将模型移动到设备
optimizer = optim.Adam(net.parameters(),lr=0.001)  #初始化迭代器
lossF = nn.MSELoss().to(device)   #初始化损失函数

EPOCHS = 30
history = []
for epoch in range(EPOCHS):
    net.train()
    runningLoss = 0.0

    for i, (cropImg, sourceImg) in tqdm(enumerate(trainData, 1)):
        cropImg, sourceImg = cropImg.to(device), sourceImg.to(device)

        # 清空梯度流
        optimizer.zero_grad()

        # 进行训练
        outputs = net(cropImg)
        loss = lossF(outputs, sourceImg)
        loss.backward()  # 反向传播
        optimizer.step()

        runningLoss += loss.item()

    averageLoss = runningLoss / i + 1
    history += [averageLoss]
    print('[INFO] Epoch %d loss: %.3f' % (epoch + 1, averageLoss))

    runningLoss = 0.0
    
# 可视化训练过程
plt.plot(history)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.savefig('/home/featurize/pythonProject5/training_loss.png')
plt.show()

# 保存模型
torch.save(net.state_dict(), '/home/featurize/pythonProject5/model.pth')
print('[INFO] Finished Training \nWuhu~')



