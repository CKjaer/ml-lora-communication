import os
import logging
from torch.utils.data import Dataset, DataLoader
import torch
from PIL import Image
import PIL
import random



class CustomImageDataset(Dataset):
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
        
        # logger.info(f"Total images after filtering by specific label {specific_label}: {len(self.img_list)}")

        # Group images by label
        label_image_dict = {}
        for img in self.img_list:
            label = int(img.split('_')[3])  # Assuming label is after 'class_'
            if label not in label_image_dict:
                label_image_dict[label] = []
            label_image_dict[label].append(img)

        # Log the number of images for each label
        # for label, images in label_image_dict.items():
        #     logger.info(f"Label: {label}, Number of images before sampling: {len(images)}")

        # Randomly sample images for each label
        
        
        self.img_list = []
        for label, images in label_image_dict.items():
            if seed!=None:
                random.seed(0)
            sampled_images = random.sample(images, min(self.samples_per_label, len(images)))
            self.img_list.extend(sampled_images)
            #logger.info(f"Label: {label}, Number of images after sampling: {len(sampled_images)}")

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
            # logger.error(f"Error loading image {img_name}: {e}")
            return None, None