import pickle
import time
import torch
from torch import nn
from torch.utils.data import DataLoader
from utils_zh.data_process import data_process
from utils_zh.model import Flower

def train():

    with open("../process_data/X_train.pkl", "rb") as f:
        X_train = pickle.load(f)

    with open("../process_data/y_train.pkl", "rb") as f:
        y_train = pickle.load(f)

    train_process = data_process(X_train, y_train)
    # 创建数据处理对象

    data = DataLoader(train_process, batch_size=32, shuffle=True)
    # 创建数据加载器 以32为一组 并且打乱顺序
    # 会自动计算要计算多少个批次的batch_size

    model = Flower(3,5)
    # 初始化模型 输入的维度定为3 因为一张图片有三个通道
    # 输出的类别为5

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    # 如果电脑有可用的cuda的话 就在cuda上运行 否则就还是使用cpu

    criterion = nn.CrossEntropyLoss()
    # 定义交叉熵损失函数

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # 使用Adam优化器

    epochs = 30
    # 设定训练轮数

    train_log = '../model/training.log'
    file = open(train_log, 'w')
    # 打印训练日志（好习惯）

    for epoch in range(epochs):

        epoch_idx = 0
        total_loss = 0
        start_time = time.time()
        # 记录每一批的训练轮数 总损失和时间

        for X,y in data:
            X, y = X.to(device), y.to(device)
            # 将输入的数据转移到上面指定的设备上

            output = model(X)
            # 前向传播

            optimizer.zero_grad()
            # 梯度清零

            loss = criterion(output, y)
            # 计算损失

            loss.backward()
            # 反向传播

            optimizer.step()
            # 更新模型参数

            total_loss += loss.item()
            # 累加每一批的损失值

            epoch_idx += 1

        message = 'Epoch:{}, Loss:{:.4f}, Time:{:.2f}'.format(epoch, total_loss / epoch_idx, time.time()-start_time)
        file.write(message + '\n')
        # 将每一次训练信息储存到日志里

        print(message)

    file.close()
    # 关闭日志文件

    torch.save(model.state_dict(), '../model/flower_model_30.pt')
    # 保存模型


