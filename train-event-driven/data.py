import os
import torch
import numpy as np
import torchvision.transforms as transforms
import random
from torch.utils.data import DataLoader
from torch.utils.data import distributed
from torchvision.datasets import CIFAR10, CIFAR100

from tonic.transforms import ToFrame
from tonic.datasets.nmnist import NMNIST
from tonic.datasets.cifar10dvs import CIFAR10DVS
from tonic.datasets.dvsgesture import DVSGesture

def DVS_Gesture_loader(root_dir, test_only=True, run_batch=False, batch_size=25, T=20, data_aug=False):
    transform_train = None
    transform_test = None
    if run_batch:
        to_raster = [
            ToFrame(sensor_size=DVSGesture.sensor_size, n_time_bins=T),
            lambda x: x.astype(np.float32),
            lambda x: torch.Tensor(x),
        ]
        if data_aug :
            data_aug = [
                lambda x: x.roll((random.randint(-20, 20), random.randint(-20, 20)), dims=[2, 3]),
            ]
        else :
            data_aug = []
        transform_train = []
        transform_test = []
        transform_train.extend(to_raster)
        transform_train.extend(data_aug)
        transform_test.extend(to_raster)
        transform_train = transforms.Compose(transform_train)
        transform_test = transforms.Compose(transform_test)
    
    if not test_only:
        train_dataset = DVSGesture(save_to=root_dir, train=True, transform=transform_train)
    test_dataset = DVSGesture(save_to=root_dir, train=False, transform=transform_test)
    
    if run_batch:
        if not test_only:
            train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=12, drop_last=True, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=12, drop_last=True, shuffle=False)

    if test_only:
        if run_batch:
            return test_loader
        else:
            return test_dataset
    else:
        if run_batch:
            return train_loader, test_loader
        else:
            return train_dataset, test_dataset

def CIFAR10DVS_loader(root_dir, test_only=True, run_batch=False, batch_size=25, T=20, data_aug=False, testset_portion=0.1):
    transform_train = None
    transform_test = None
    if run_batch:
        to_raster = [
            ToFrame(sensor_size=CIFAR10DVS.sensor_size, n_time_bins=T),
            lambda x: x.astype(np.float32),
            lambda x: torch.Tensor(x),
        ]
        if data_aug :
            # rand_hflip = transforms.RandomVerticalFlip(p=0.5)
            data_aug = [
                # lambda x: rand_hflip(x),
                lambda x: x.flip(dims=[2]) if random.randint(0, 1) > 0.5 else x,
                # lambda x: print(x.shape),
                lambda x: x.roll((random.randint(-20, 20), random.randint(-20, 20)), dims=[2, 3])
            ]
        else :
            data_aug = []
        transform_train = []
        transform_test = []
        transform_train.extend(to_raster)
        transform_train.extend(data_aug)
        transform_test.extend(to_raster)
        transform_train = transforms.Compose(transform_train)
        transform_test = transforms.Compose(transform_test)
    
    train_dataset = CIFAR10DVS(save_to=root_dir, transform=transform_train)

    if not test_only :
        new_data = []
        new_targets = []
        for i in range(10) : ## Split dataset
            for j in range(i * 1000, round((i + 1) * 1000 - testset_portion * 1000)) :
                new_data.append(train_dataset.data[j])
                new_targets.append(train_dataset.targets[j])
        train_dataset.data = new_data
        train_dataset.targets = new_targets

    test_dataset = CIFAR10DVS(save_to=root_dir, transform=transform_test)
    new_data = []
    new_targets = []
    for i in range(10) : ## Split dataset
        for j in range(round((i + 1) * 1000 - testset_portion * 1000), (i + 1) * 1000) :
            new_data.append(test_dataset.data[j])
            new_targets.append(test_dataset.targets[j])
    test_dataset.data = new_data
    test_dataset.targets = new_targets

    
    if run_batch:
        if not test_only:
            train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=12, drop_last=True, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=12, drop_last=True, shuffle=False)

    if test_only:
        if run_batch:
            return test_loader
        else:
            return test_dataset
    else:
        if run_batch:
            return train_loader, test_loader
        else:
            return train_dataset, test_dataset


def save_aedat4_dataset_as_np_format(dataset, save_dir, save_type=np.int32) :
    for i in range(len(dataset.data)) :
        name = os.path.basename(dataset.data[i])
        x = dataset[i]
        category = x[1]

        x = x[0]
        x = np.stack([x['t'].astype(save_type), x['x'].astype(save_type), x['y'].astype(save_type), x['p'].astype(save_type)], axis=1)
        # ret = []
        # for i in range(x.shape[0]) :
        #     ret.append([int(x[i][j]) for j in range(4)])
        # ret = np.stack(ret)
        cat_dir = os.path.join(save_dir, str(category))
        if not os.path.exists(cat_dir) :
            os.mkdir(cat_dir)
        np.save(os.path.join(cat_dir, name + '.npy'), x)


def cifar_loader(batch_size=128, cutout=False, workers=4, use_cifar10=False, dpath='./datasets', resize224=False, DDP=False):

    aug = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
    tr_test = []
    if cutout:
        aug.append(Cutout(n_holes=1, length=16))
    if resize224:
        aug.append(transforms.Resize((224, 224)))
        tr_test.append(transforms.Resize((224, 224)))
    aug.append(transforms.ToTensor())
    tr_test.append(transforms.ToTensor())


    if use_cifar10:
        aug.append(transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))
        tr_test.append(transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))
        transform_train = transforms.Compose(aug)
        transform_test = transforms.Compose(tr_test)
        train_dataset = CIFAR10(root=os.path.join(dpath, 'cifar10'),
                                train=True, download=True, transform=transform_train)
        val_dataset = CIFAR10(root=os.path.join(dpath, 'cifar10'),
                              train=False, download=True, transform=transform_test)

    else:
        aug.append(transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),)
        tr_test.append(transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)))
        transform_train = transforms.Compose(aug)
        transform_test = transforms.Compose(tr_test)
        train_dataset = CIFAR100(root=os.path.join(dpath, 'cifar100'),
                                 train=True, download=True, transform=transform_train)
        val_dataset = CIFAR100(root=os.path.join(dpath, 'cifar100'),
                               train=False, download=True, transform=transform_test)

    train_sampler = distributed.DistributedSampler(train_dataset,shuffle=True) if DDP else None
    val_sampler = distributed.DistributedSampler(val_dataset,shuffle=False) if DDP else None
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True if not DDP else None,
                            num_workers=workers, pin_memory=True, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False if not DDP else None, num_workers=workers, pin_memory=True, sampler=val_sampler)

    return train_loader, val_loader


if __name__ == '__main__' :
    _, test_loader = CIFAR10DVS_loader("/data_smr/dataset", False, False)
    save_aedat4_dataset_as_np_format(test_loader, "/data_smr/dataset/CIFAR10DVS_PTEVENT")