# 2014 VGG16
# 2个3*3卷积核 等同于 1个5*5卷积核
# 小卷积核组合 优于 大卷积核
# 验证了加深网络是可行的
import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# Hyper Parameters
EPOCH = 1  # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 50
LR = 0.001  # learning rate


# data

class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        # input 224*224*3
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # conv1 112*112*64
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # conv2 56*56*128
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # conv3 28*28*256
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # conv4 14*14*512
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # conv5 7*7*512
        self.classifier = nn.Sequential(
            nn.Linear(512 * 512 * 3, 4096),
            nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 1000)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(x.size(0), -1) # 将多维度的Tensor展平成一维
        y = self.classifier(x)
        return y


if __name__ == '__main__':
    vgg16 = VGG16()
    print(vgg16)


