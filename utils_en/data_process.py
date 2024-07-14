from torch.utils.data import Dataset

class data_process(Dataset):

    def __init__(self, X, y):
        self.X = X
        self.y = y
    # Initialize

    def __len__(self):
        return len(self.X)
    # Return the length of the data; the dataloader will use this length
    # to automatically calculate how many epochs are needed for training

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    # Since the first dimension is the number of stacked tensors,
    # we can directly use the first dimension's index to fetch each image [N, C, H, W]
