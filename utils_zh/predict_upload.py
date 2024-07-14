import pickle
import torchvision
import torch
from PIL import Image
from utils_zh.model import Flower

def predict_upload(image_path):

    with open('../process_data/flower_labels.pkl','rb') as f:
        labels = pickle.load(f)

    file = image_path
    # 这里如果不想用GUI的话 把image_path改成图片的地址即可

    image = Image.open(file).convert('RGB')
    # 将路径上的图片加载并转换为RGB格式

    image_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),\
                                                     torchvision.transforms.Resize((227, 227))])
    # 定义一系列操作 再后续直接应用到图片处理中
    # 将输入的图片压缩或扩展为227*227的格式 并转化为张量形式

    image = image_transform(image)
    # 因为只有一张图片且没有标签 所以直接转换格式就行 不用ImageFolder

    image = image.unsqueeze(0)
    # 因为输入的格式前面要有一个批次 所以在前面曾加一个维度
    # [1, 3, 227, 227]

    model = Flower(3,5)
    model.load_state_dict(torch.load('../model/flower_model_30.pt',map_location=torch.device('cpu')))
    model.eval()
    # 构建模型 加载模型 设置评估模式

    with torch.no_grad():
        # 禁用梯度计算

        output = model(image)
        # 向前传播

        _, predicted = torch.max(output, 1)
        # 返回五个类别中的概率最大值的索引

        print(f'predicted: {labels[predicted.item()]}')

        return labels[predicted.item()]
        # 通过data_loader文件中保存的字典取出来最后的答案并返回
