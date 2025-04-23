import os
import torchvision.transforms as transforms
from .autoaugment import CIFAR10Policy, Cutout
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100


def build_data(batch_size=128, cutout=False, workers=4, use_cifar10=False, auto_aug=False, dpath='./datasets'):

    aug = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
    if auto_aug:
        aug.append(CIFAR10Policy())

    aug.append(transforms.ToTensor())

    if cutout:
        aug.append(Cutout(n_holes=1, length=16))

    if use_cifar10:
        aug.append(
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))
        transform_train = transforms.Compose(aug)
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_dataset = CIFAR10(root=os.path.join(dpath, 'cifar10/'),
                                train=True, download=True, transform=transform_train)
        val_dataset = CIFAR10(root=os.path.join(dpath, 'cifar10/'),
                              train=False, download=True, transform=transform_test)

    else:
        aug.append(
            transforms.Normalize(
                (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        )
        transform_train = transforms.Compose(aug)
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        train_dataset = CIFAR100(root=os.path.join(dpath, 'cifar100'),
                                 train=True, download=True, transform=transform_train)
        val_dataset = CIFAR100(root=os.path.join(dpath, 'cifar100'),
                               train=False, download=True, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=workers, pin_memory=True)

    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=workers, pin_memory=True)

    return train_loader, val_loader

def build_normal_data(batch_size=128, use_cifar10=False, dpath='./datasets', download=True, workers=4, aug=True) :
    if aug :
        aug = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
    else :
        aug = []
    aug.append(transforms.ToTensor())

    if use_cifar10:
        aug.append(
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), )
        transform_train = transforms.Compose(aug)
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_dataset = CIFAR10(root=os.path.join(dpath, 'cifar10'),
                                train=True, download=download, transform=transform_train)
        val_dataset = CIFAR10(root=os.path.join(dpath, 'cifar10'),
                              train=False, download=download, transform=transform_test)

    else:
        aug.append(
            transforms.Normalize(
                (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        )
        transform_train = transforms.Compose(aug)
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        train_dataset = CIFAR100(root=os.path.join(dpath, 'cifar100'),
                                 train=True, download=download, transform=transform_train)
        val_dataset = CIFAR100(root=os.path.join(dpath, 'cifar100'),
                               train=False, download=download, transform=transform_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=workers, pin_memory=True)

    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=workers, pin_memory=True)

    return train_loader, val_loader