import os
import logging
from torch.utils.data import Dataset, DataLoader
import torch
from PIL import Image
import PIL
import random

class CustomIQDataset(Dataset):
    def __init__(self, data_dir, snr=None, rate_param=None, transform=None, samples_per_label=250, seed=None):
        self.data_dir = data_dir
        self.data_list = os.listdir(data_dir)
        self.transform = transform
        self.snr = snr
        self.rate_param = rate_param
        self.samples_per_label = samples_per_label

        
        if self.snr is not None:
            self.data_list = [data for data in self.data_list if float(data.split('_')[1]) == self.snr] # filter to load only current snr

        if self.rate_param is not None:
            self.data_list = [data for data in self.data_list if float(data.split('_')[5]) == self.rate_param] # then filter to keep only current rate within current snr
        
        # grab only specified amount of samples per label
        if seed!=None:
            random.seed(seed)
        else:
            random.seed(0)
        self.data_list = random.sample(self.data_list, min(self.samples_per_label, len(self.data_list)))
        
        # attach label (real symbol value) to the data_list
        for data in self.data_list:
            label = int(data.split('_')[3])
            data = (data, label) # tuple of filename and label
            print("here")
            


        # # Create a dictionary of labels (symbols) and corresponding data files
        # data_labels_dict = {}
        # for data in self.data_list:
        #     label = int(data.split('_')[3]) # grab the label from the filename
        #     if label not in data_labels_dict:
        #         data_labels_dict[label] = [] # create new field in dictionary if label not present
        #     data_labels_dict[label].append(data) # store data file in corresponding label field
        
        
        # # Sample a fixed number of images per label (useful if we only want to use part of the dataset)
        # self.data_list = [] #TODO very inefficient to clear after grabbing labels, but it works for now
        # for label, data in data_labels_dict.items():
        #     if seed!=None:
        #         random.seed(0)
        #     sampled_data = random.sample(data, min(self.samples_per_label, len(data))) # get specified number of samples, or entire dataset if smaller than specified amount
        #     self.data_list.extend(sampled_data)


    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data_file_name = self.data_list[idx]
        input_path = os.path.join(self.data_dir, data_file_name)

        try:
            # Load image and extract label from filename
            image = Image.open(img_path).convert("L")  # Convert to grayscale if necessary
            label = int(img_name.split('_')[3])

            # Apply transformations
            if self.transform:
                image = self.transform(image)

            label = torch.tensor(label, dtype=torch.long)

            return image, label

        except (PIL.UnidentifiedImageError, IndexError, FileNotFoundError) as e:
            # logger.error(f"Error loading image {img_name}: {e}")
            return None, None
        
if __name__ == "__main__":
    dataset = CustomIQDataset(data_dir="output/20241114-093838/csv", snr=-6, rate_param=0.5, samples_per_label=250, seed=0)