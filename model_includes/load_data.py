import sys
import os
from torchvision import transforms
from torch.utils.data import random_split, DataLoader, Dataset
import torch
import logging
from PIL import Image
import PIL
import random


class CustomImageDataset(Dataset):
    """
    A custom dataset class for loading images from a directory, with optional filtering and sampling.
    Args:
        img_dir (str): Directory containing the images.
        specific_label (float, optional): Specific label to filter images by. Defaults to None.
        rate_param (float, optional): Specific rate parameter to filter images by. Defaults to None.
        transform (callable, optional): A function/transform to apply to the images. Defaults to None.
        samples_per_label (int, optional): Number of samples to take per label. Defaults to 1000000.
        seed (int, optional): Random seed for reproducibility. Defaults to None.
    Attributes:
        img_dir (str): Directory containing the images.
        img_list (list): List of image filenames after filtering and sampling.
        transform (callable): A function/transform to apply to the images.
        specific_label (float): Specific label to filter images by.
        rate_param (float): Specific rate parameter to filter images by.
        samples_per_label (int): Number of samples to take per label.
    """

    def __init__(self, img_dir, specific_label=None, rate_param=None, transform=None, samples_per_label=1000000, seed=None):
        self.img_dir = img_dir
        self.img_list = os.listdir(img_dir)
        self.transform = transform
        self.specific_label = specific_label
        self.rate_param = rate_param
        self.samples_per_label = samples_per_label

        # Filter images based on specific label (if provided)
        if specific_label is not None:
            self.img_list = [img for img in self.img_list if float(img.split('_')[1]) == specific_label]

        if rate_param is not None:
            self.img_list = [img for img in self.img_list if float(img.split('_')[5]) == rate_param]
        
        # Group images by label
        label_image_dict = {}
        for img in self.img_list:
            label = int(img.split('_')[3])  # Assuming label is after 'class_'
            if label not in label_image_dict:
                label_image_dict[label] = []
            label_image_dict[label].append(img)

        # Randomly sample images for each label
        self.img_list = []
        for label, images in label_image_dict.items():
            if seed!=None:
                random.seed(0)
            sampled_images = random.sample(images, min(self.samples_per_label, len(images)))
            self.img_list.extend(sampled_images)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_name = self.img_list[idx]
        img_path = os.path.join(self.img_dir, img_name)

        try:
            # Load image and extract label from filename
            image = Image.open(img_path).convert("L")  # Convert to grayscale if necessary
            label = int(img_name.split('_')[3])  # Assuming label is after 'class_'

            # Apply transformations
            if self.transform:
                image = self.transform(image)

            label = torch.tensor(label, dtype=torch.long)

            return image, label

        except (PIL.UnidentifiedImageError, IndexError, FileNotFoundError) as e:
            return None, None


def load_data(data_dir, training:bool, batch_size, SNR, rate_param, img_size:list, M=2**7):
    """
    Load data for training and perform 80/20 validation split.
    Args:
        data_dir (str): Directory containing the data.
        training (bool): Flag indicating whether to load data for training or testing.
        batch_size (int): Number of samples per batch.
        SNR (int): Signal-to-noise ratio label for filtering the dataset.
        rate_param (float): Rate arrival parameter
        img_size (list): List containing the desired image size [height, width].
        M (int, optional): Modulation order, default is 2^7.
    Returns:
        DataLoader: DataLoader object for training and testing datasets if training is True.
        DataLoader: DataLoader object for the entire dataset if training is False.
    """

    # Transform images and convert to tensor
    transform = transforms.Compose([
        transforms.Resize((img_size[0], img_size[1])),  # Resize image
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize with mean and std
    ])

    dataset = CustomImageDataset(img_dir=data_dir, specific_label=SNR, rate_param=rate_param, transform=transform)
    
    # Load data for either training or testing
    if training==True:
        # Split dataset into training and validation sets with a 80/20 split
        train_size = int(len(dataset) * 0.8)
        validation_size = len(dataset) - train_size

        # Randomize the split
        train_dataset, val_dataset = random_split(dataset, [train_size, validation_size])
        
        # Load the data
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader
    else:
        # Load the entire dataset for testing
        test_loader = DataLoader(dataset, shuffle=True)
        return test_loader