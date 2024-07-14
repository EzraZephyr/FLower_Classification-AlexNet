import pickle
import torch
from torch.utils.data import DataLoader
from utils_zh.model import Flower
from utils_zh.data_process import data_process

def predict():
    with open('../process_data/X_test.pkl','rb') as f:
        X_test = pickle.load(f)

    with open('../process_data/y_test.pkl','rb') as f:
        y_test = pickle.load(f)

    test_process = data_process(X_test, y_test)
    # 创建数据处理对象

    data_loader = DataLoader(test_process, batch_size=1, shuffle=False,)
    # 创建数据加载器

    model = Flower(3,5)
    # 构建模型

    model.load_state_dict(torch.load('../model/flower_model_30.pt',map_location=torch.device('cpu')))
    # 加载模型
    # 因为模型训练用的是cuda 所以测试如果不是cuda的话 需要用map_location转到cpu

    model.eval()
    # 设置为评估模式

    total = 0
    correct = 0

    with torch.no_grad():
        # 禁用梯度计算

        for X, y in data_loader:
            output = model(X)
            # 向前传播

            _, pred = torch.max(output, 1)
            # 返回五个类别中最大值及其索引

            total += y.size(0)
            # 计算样本总数

            correct += (pred == y).sum().item()
            # 计算预测正确的样本总数

    print(f'Accuracy:{100 * correct / total}%')


