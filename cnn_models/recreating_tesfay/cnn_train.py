import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import PIL
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
import matplotlib.pyplot as plt
import logging
import numpy as np
import random
from numpy import savetxt
from scipy import stats
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Directory paths
img_dir = "./output/20241101-121700/plots"
output_folder = './cnn_output/snrScaling'
models_folder = os.path.join(output_folder, 'models')

# Create the directory if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    os.makedirs(models_folder)
    
# Configure logger
logfilename = "cnn.log"
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',datefmt='%Y-%m-%d %H:%M:%S', filename=os.path.join(output_folder, logfilename), encoding='utf-8', level=logging.INFO)
logger.info("Starting the program")


# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Define the CNN architecture
class LoRaCNN(nn.Module):
    def __init__(self, M):
        super(LoRaCNN, self).__init__()
        
        # Convolutional layers with batch normalization
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=M//4, kernel_size=4, stride=1, padding=2)
        self.conv2 = nn.Conv2d(in_channels=M//4, out_channels=M//2, kernel_size=4, stride=1, padding=2)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.batchNorm1 = nn.BatchNorm2d(M//4)
        self.batchNorm2 = nn.BatchNorm2d(M//2)
        
        # Fully connected layers with batch normalization
        self.fc1 = nn.Linear(M//2 * (M//4) * (M//4), 4 * M)
        self.fc2 = nn.Linear(4 * M, 2 * M)
        self.fc3 = nn.Linear(2 * M, 128)  # Output size is 128 for 128 labels
        self.batchNorm3 = nn.BatchNorm1d(4 * M)
        self.batchNorm4 = nn.BatchNorm1d(2 * M)
        
    def forward(self, x):
        # Convolutional layers
        x = F.relu(self.batchNorm1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.batchNorm2(self.conv2(x)))
        x = self.pool(x)
        
        # Flatten the output from conv layers
        x = x.view(-1, self.num_flat_features(x))
        
        # Fully connected layers
        x = F.relu(self.batchNorm3(self.fc1(x)))
        x = F.relu(self.batchNorm4(self.fc2(x)))
        x = self.fc3(x)
        
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # All dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
class CustomImageDataset(Dataset):
    def __init__(self, img_dir, specific_label=None, rate_param=None, transform=None, samples_per_label=250):
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
        
        logger.info(f"Total images after filtering by specific label {specific_label}: {len(self.img_list)}")

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
            logger.error(f"Error loading image {img_name}: {e}")
            return None, None

 

# Define transformations for the images
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize images to a fixed size
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize with mean and std
])


# Training function
def train(model, train_loader, num_epochs, optimizer, criterion):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if i % 100 == 99:
                logger.info(f'Epoch [{epoch+1}], Step [{i+1}], Loss: {running_loss / 100:.4f}')
                running_loss = 0.0


# Evaluation function
def evaluate_and_calculate_ser(model, test_loader, criterion):
    model.eval()  # Set model to evaluation mode
    correct_predictions = 0
    total_predictions = 0
    total_loss = 0.0
    incorrect_predictions = 0

    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            correct_predictions += (predicted == labels).sum().item()
            incorrect_predictions += (predicted != labels).sum().item()
            total_predictions += labels.size(0)

    accuracy = 100 * correct_predictions / total_predictions
    ser = incorrect_predictions / total_predictions
    average_loss = total_loss / len(test_loader)

    logger.info(f'Validation/Test Loss: {average_loss:.4f}')
    logger.info(f'Validation/Test Accuracy: {accuracy:.2f}%')
    logger.info(f'Symbol Error Rate (SER): {ser:.6f}')
    return ser

# List of snr and rate parameters for which SER will be calculated
snr_list = [i for i in range(-16, -2, 2)] # TODO change this to -16, -2, 2
rates = [0, 0.25, 0.5, 0.7, 1] 

# Placeholder to store symbol error rates
symbol_error_rates = {} # dictionary to store SER for each rate
for rate in rates:
    symbol_error_rates[rate] = []


# Hyperparameters
M = 128  # Number of classes
batch_size = 32
learning_rate = 0.02
num_epochs = 3
optimizer_choice = 'SGD' # 'Adam' or 'SGD'
criterion = nn.CrossEntropyLoss()

# Loop over each specific value
for snr in snr_list:
    for rate in rates:
        logger.info(f"Calculating SER for snr: {snr}, rate {rate}")

        dataset = CustomImageDataset(img_dir=img_dir, specific_label=float(snr), rate_param=float(rate), transform=transform, samples_per_label=250)
        logger.info(f"Number of images in dataset: {len(dataset)}")
        # Dataset size check
        if len(dataset) == 0:
            logger.warning(f"No images found for specific value: {snr}. Skipping.")
            continue

        dataset_size = len(dataset)
        train_size = int(0.8 * dataset_size)
        test_size = dataset_size - train_size

        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        model = LoRaCNN(M).to(device)
        
        if optimizer_choice == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        elif optimizer_choice == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=learning_rate)
            

        # Train the model
        train(model, train_loader, num_epochs, optimizer, criterion)

        # Save model and optimizer
        torch.save(model.state_dict(), os.path.join(models_folder, f'model_snr_{snr}_rate{rate}.pth'))
        torch.save(optimizer.state_dict(), os.path.join(models_folder, f'optimizer_snr_{snr}_rate_{rate}.pth'))

        # Evaluate model and calculate SER
        ser = evaluate_and_calculate_ser(model, test_loader, criterion)
        symbol_error_rates[rate].append((ser, snr)) # store SER and SNR value in corresponding rate
        


logger.info("All SER values have been calculated.")


# parameters for plotting
# plt.rcParams['mathtext.fontset'] = 'custom'
# plt.rcParams['mathtext.rm'] = 'TeX Gyre Pagella'
# plt.rcParams['font.family'] ='TeX Gyre Pagella'
fs = 20
plt.rcParams.update({'font.size': fs})


for rate, values in symbol_error_rates.items():
    snr_values = list(map(int, values.keys()))
    ser_values = list(values.values())
    snr_values = sorted(snr_values)
    
    if rate == 0:
        zero_snr_values = snr_values
        zero_ser_values = ser_values

    savetxt(f'snr_vs_ser_rate_{rate}.csv', np.array([snr_values, ser_values]).T, delimiter=';', fmt='%d;%.6f')
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(
        zero_snr_values,
        zero_ser_values,
        label=f"SF:{7} λ=0.00",
        linestyle="dashed",
        color="black",
        marker="v"
    )
    ax.plot(
        snr_values,
        ser_values,
        marker="v",
        color="black",
        label=f"SF:{7} λ={rate:.2f}"
    )
    ax.set_xlabel('SNR [dB]')
    ax.set_ylabel('SER')
    ax.set_yscale('log')
    ax.set_ylim(1e-5, 1)
    ax.set_xlim(-16, -4)
    ax.grid(True, which="both", alpha=0.5)
    ax.legend(loc='upper right')
    
    if rate != 0:
        # Create an inset with the Poisson PMF stem plot
        inset_ax = inset_axes(
            ax,
            width="30%",
            height="40%",
            loc="lower left",
            bbox_to_anchor=(0.1, 0.1, 1, 1),
            bbox_transform=ax.transAxes,
        )
        l = np.linspace(0,10,11)
        poisson_dist = stats.poisson.pmf(l, mu=rate)
        print(poisson_dist)
        mask = (poisson_dist >= 0.005)
        inset_ax.set_title(f"PMF, λ={rate:.2f}", fontsize = (fs - 2))
        inset_ax.set_xlabel(r"$\mathrm{N_i}$", labelpad=-4, fontsize = (fs - 2))
        inset_ax.set_xlim([0, 10])
        inset_ax.set_ylim([0, 0.8])
        inset_ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8])

        stem_inset = inset_ax.stem(
            l[mask],
            poisson_dist[mask],
            basefmt=" ",
            linefmt="k-",
        )
        # Allow clipping of the stem plot
        for artist in stem_inset.get_children():
            artist.set_clip_on(False)

    plt.tight_layout()
        
    # Save the plot
    plot_filename = os.path.join(output_folder, f'snr_vs_ser_rate_{rate}.pdf')
    plt.savefig(plot_filename)
    plt.show()
    plt.close()
    logger.info(f"Plot for rate {rate} has been saved to {plot_filename}.")
    

#ser_values_dashed_circle = np.array([1.0, 0.16, 0.13, 0.02, 0.003, 3.16e-5, 0])

# Plotting SNR vs SER
# plt.figure(figsize=(10, 6))
# plt.plot(specific_values, symbol_error_rates, marker='o', linestyle='-', color='b')
# #plt.plot(specific_values, ser_values_dashed_circle, marker='o', linestyle='--', color='black', label='CNN output in paper')
# plt.xlabel('SNR')
# plt.ylabel('Symbol Error Rate (SER)')
# plt.yscale('log')
# plt.ylim(1e-5, 1e0)
# plt.title('SNR vs Symbol Error Rate')
# plt.legend(['CNN Output', 'CNN Output in Paper'])
# plt.grid(True)

# # Save and show plot
# final_plot_filename = os.path.join(output_folder, 'snr_vs_ser_final_plot.png')
# plt.savefig(final_plot_filename)
# plt.show()
# plt.close()

# print(f"Final plot has been saved to {final_plot_filename}.")
