from imageDataSet import CustomImageDataset
from torchvision import transforms
from torch.utils.data import random_split, DataLoader
import torch


def loadData(img_dir, batch_size, SNR, rate_param, M=2**7, seed=None):

    transform = transforms.Compose([
        transforms.Resize((M, M)),  # Resize images to a fixed size
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize with mean and std
    ])


    if seed!=None:
        seed=0
        generator=torch.Generator().manual_seed(seed)
        dataset =CustomImageDataset(img_dir=img_dir, specific_label=SNR, rate_param=rate_param, transform=transform, seed=seed)
        train_size=int(len(dataset)*0.8)
        test_size=len(dataset)-train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=generator)
    else:
        dataset =CustomImageDataset(img_dir=img_dir, specific_label=SNR, rate_param=rate_param, transform=transform)
        train_size=int(len(dataset)*0.8)
        test_size=len(dataset)-train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader