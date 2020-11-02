import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose

class CustomDataset(Dataset):

    def __init__(self, transform=None):
        data = np.ones((1000, 5), dtype=np.float32)
        self.n_samples = data.shape[0]

        self.x = data[:, 1:]
        self.y = data[:, [0]]

        self.transform = transform

    def __getitem__(self, index):
        sample = self.x[index], self.y[index]

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return self.n_samples

class ToTensor:
    def __call__(self, sample):
        x, y = sample
        return torch.from_numpy(x), torch.from_numpy(y)

class ToGPU:
    def __call__(self, sample):
        x, y = sample
        return x.to(device), y.to(device)

class MulTransform:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, sample):
        x, y = sample
        x *= self.factor
        return x, y

device = torch.device("cuda:0")
compose = Compose([
    ToTensor(),
    MulTransform(2),
    ToGPU()
])

dataset = CustomDataset(transform=compose)
x, y = dataset[0]
print(x, y)
print(x.shape, y.shape)

dataloader = DataLoader(dataset,
                        batch_size=4,
                        shuffle=True)

b_x, b_y = next(iter(dataloader))
print(b_x, b_y)
print(b_x.shape, b_y.shape)

