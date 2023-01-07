from multiprocessing import cpu_count

import torch
import torchvision
import torchvision.transforms as transforms


def get_train_dataloader(batch_size, num_workers = None):
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding = 4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    training_set = torchvision.datasets.CIFAR10(
        root='./data', train = True, download = True, transform = transform_train)

    train_dataloader = torch.utils.data.DataLoader(
        training_set, batch_size = batch_size, shuffle = True, num_workers = num_workers if num_workers is not None else cpu_count())
    
    return train_dataloader


def get_test_dataloader(batch_size, num_workers = 4):

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    testing_set = torchvision.datasets.CIFAR10(
        root='./data', train = False, download = True, transform = transform_test)

    test_dataloader = torch.utils.data.DataLoader(
        testing_set, batch_size = batch_size, shuffle = False, num_workers = num_workers)

    return test_dataloader