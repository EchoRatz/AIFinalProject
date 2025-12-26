import torch
from torchvision import datasets, transforms

def get_cifar10():
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ]) 
    
    train_set = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform_train
    )
    
    test_set = datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transform_test
    )
    
    return train_set, test_set