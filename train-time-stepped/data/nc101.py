import os
import torch
import random
import torchvision.transforms as transforms
from torch.utils.data import Dataset

# 准备数据集并预处理
class NCaltech101(Dataset):
    def __init__(self, root, train=True, transform=False, target_transform=False, preload = False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.resize = transforms.Resize(size=(48, 48))  # 48 48
        self.tensorx = transforms.ToTensor()
        self.imgx = transforms.ToPILImage()
        self.preloaded = preload
        if preload:
            self.preload()

    def __getitem__(self, index):
        if self.preloaded :
            data, target = self.data[index]
        else : data, target = self.load(index)
        new_data = []
        for t in range(data.shape[0]):
            new_data.append(self.tensorx(self.resize(self.imgx(data[t, ...]))))
        data = torch.stack(new_data, dim=0)
        if self.transform:
            flip = random.random()
            if flip > 0.75:
                data = torch.flip(data, dims=(3,))
            if flip < 0.25:
                data = torch.flip(data, dims=(0,1))
            off1 = random.randint(-5, 5)
            off2 = random.randint(-5, 5)
            data = torch.roll(data, shifts=(off1, off2), dims=(2, 3))

        if self.target_transform:
            target = self.target_transform(target)
        return data, target;

    def __len__(self):
        return len(os.listdir(self.root))

    def load(self, index) :
        data, target = torch.load(self.root + '/{}.pt'.format(index))
        return data, target;

    def preload(self) :
        self.data = [];
        for i in range(self.__len__()) :
            self.data.append(self.load(i))


def build_data(batch_size, path) :
    train_dataset = NCaltech101(path + "/train", transform=True, preload=True)
    test_dataset = NCaltech101(path + "/test", transform=False, preload=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader