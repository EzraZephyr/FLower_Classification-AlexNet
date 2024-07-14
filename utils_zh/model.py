from torch import nn


class Flower(nn.Module):

    def __init__(self, in_dim,n_class):
        super(Flower, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, 96, 11, 4, padding=0),
            nn.ReLU(True),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(96, 256, 5, 1, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(256, 384, 3, 1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(384, 384, 3, 1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(384, 256, 3, 1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(3, 2),
            # 定义AlexNet方法中的卷积层和池化层 中间包含Relu函数
            # 可以上网搜索“AlexNet 结构”来详细了解

        )

        self.fc = nn.Sequential(
            nn.Linear(9216, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, n_class),
            # 定义两个全连接层和输出层 中间包含Relu函数
            # 可以上网搜索“AlexNet 结构”来详细了解

        )



    def forward(self, input):
        X = self.conv(input)
        # 将输入的数据通过定义好的卷积层和池化层

        X = X.view(X.size(0), -1)
        # 将得到的数据的二三四层相乘 也就是展平 准备输入全连接层

        X = self.fc(X)
        # 输入全连接层

        return X

