from torch.utils.data import Dataset


class data_process(Dataset):

    def __init__(self, X, y):
        self.X = X
        self.y = y
    # 初始化

    def __len__(self):
        return len(self.X)
    # 返回数据的长度 dataloader会根据这个长度
    # 自动计算出需要训练多少轮

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    # 因为之前在加载数据的时候 第一维是我们堆叠的张量的数量
    # 所以可以直接通过第一维的索引取出每一份图片 [N, C, H, W]