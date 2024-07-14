import torchvision
import torch
import pickle
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split

def data_loader():

    flower_image = '../data/flower_images/'

    dataset_transform = torchvision.transforms.Compose([torchvision.transforms.Resize((227, 227)),
                                                        torchvision.transforms.ToTensor(),])
    # 定义一系列操作 再后续直接应用到图片处理中
    # 将输入的图片压缩或扩展为227*227的格式 并转化为张量形式


    flower_data = ImageFolder(root=flower_image, transform=dataset_transform)
    # 将上面定义的操作通过ImageFolder部署到子文件的每一张图片中 并且将子文件的名称
    # 也就是图片的类别从0开始转换为数值 并储存为字典形式

    class_to_idx = flower_data.class_to_idx
    index_to_class = {idx:cla for cla, idx in class_to_idx.items()}
    # 以下为通过ImageFolder将子文件转换为的字典形式
    # {'Lilly': 0, 'Lotus': 1, 'Orchid': 2, 'Sunflower': 3, 'Tulip': 4}
    # 需要将value转换为key 这样在后面预测函数的时候
    # 可以直接输出花卉的名称 而非数字类别



    with open('../process_data/flower_labels.pkl', 'wb') as f:
        pickle.dump(index_to_class, f)

    X, y = [],[]
    i = 0
    for image, label in flower_data:
        i+=1
        if i % 1000 == 0:
            print(i)
        # 观测数据处理进度

        X.append(image)
        y.append(label)
        # 将取出的图片张量和其目标值进行存放

    X = torch.stack(X)
    # 将所有在列表中的张量 延新维度‘堆叠’成为一个新的张量列表
    # (N,...)

    y = torch.tensor(y)
    # 将目标值转换为张量

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # 划分训练集和数据集

    with open('../process_data/X_train.pkl', 'wb') as f:
        pickle.dump(X_train, f)
    with open('../process_data/X_test.pkl', 'wb') as f:
        pickle.dump(X_test, f)
    with open('../process_data/y_train.pkl', 'wb') as f:
        pickle.dump(y_train, f)
    with open('../process_data/y_test.pkl', 'wb') as f:
        pickle.dump(y_test, f)
    # 保存数据
