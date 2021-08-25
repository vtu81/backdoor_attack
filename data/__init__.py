from .poisoned_dataset import PoisonedDataset
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
import torchvision


def load_init_data(dataname, device, download, dataset_path):
    if dataname == 'mnist':
        train_data = datasets.MNIST(root=dataset_path, train=True, download=download)
        test_data  = datasets.MNIST(root=dataset_path, train=False, download=download)
    elif dataname == 'cifar10':
        train_data = datasets.CIFAR10(root=dataset_path, train=True,  download=download)
        test_data  = datasets.CIFAR10(root=dataset_path, train=False, download=download)
    return train_data, test_data


def create_backdoor_data_loader(dataname, train_data, test_data, trigger_label, poisoned_portion, batch_size, device, mark_dir=None, alpha=1.0):
    train_data    = PoisonedDataset(train_data, trigger_label, portion=poisoned_portion, mode="train", device=device, dataname=dataname, mark_dir=mark_dir, alpha=alpha, train=True)
    test_data_ori = PoisonedDataset(test_data,  trigger_label, portion=0,                mode="test",  device=device, dataname=dataname, mark_dir=mark_dir, alpha=alpha, train=False)
    test_data_tri = PoisonedDataset(test_data,  trigger_label, portion=1,                mode="test",  device=device, dataname=dataname, mark_dir=mark_dir, alpha=alpha, train=False)

    if device == torch.device("cpu"):
        train_data_loader       = DataLoader(dataset=train_data,    batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
        test_data_ori_loader    = DataLoader(dataset=test_data_ori, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
        test_data_tri_loader    = DataLoader(dataset=test_data_tri, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    else:
        train_data_loader       = DataLoader(dataset=train_data,    batch_size=batch_size, shuffle=True)
        test_data_ori_loader    = DataLoader(dataset=test_data_ori, batch_size=batch_size, shuffle=True)
        test_data_tri_loader    = DataLoader(dataset=test_data_tri, batch_size=batch_size, shuffle=True)

    return train_data_loader, test_data_ori_loader, test_data_tri_loader
