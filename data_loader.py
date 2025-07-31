import torch 
import torchvision.transforms as transforms 

from torchvision import datasets
from torch.utils.data import DataLoader


transform = transforms.Compose([transforms.ToTensor()]) 


def get_loader(batch_size=64, data_dir='./data'):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = datasets.MNIST(root=data_dir, train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root=data_dir, train=False, transform=transform, download=True)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader