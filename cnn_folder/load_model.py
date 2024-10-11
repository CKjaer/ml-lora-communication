import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from cnn_v1 import LoRaCNN # import CNN architecture from cnn_v1.py
import argparse
import logging
import PIL
import json
import matplotlib as plt

def load_model(model_path, M=128):
    # Load the model
    model = LoRaCNN(M)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

class CustomImageDataset(Dataset):
    def __init__(self, img_dir, specific_label=None, transform=None):
        self.img_dir = img_dir
        self.img_list = os.listdir(img_dir)
        self.transform = transform
        self.specific_label = specific_label

        if specific_label is not None:
            self.img_list = [img for img in self.img_list if float(img.split('_')[1]) == specific_label]
    
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
            # If an error occurs, we can either return None or handle it as appropriate.
            return None, None

# Define transformations for the images
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize images to a fixed size
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize with mean and std
])

def evaluate_and_calculate_ser(model, test_loader, criterion):
    correct_predictions = 0
    total_predictions = 0
    total_loss = 0.0
    incorrect_predictions = 0

    with torch.no_grad():  # Disable gradient calculation for evaluation
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU if available
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            # Get predicted labels
            _, predicted = torch.max(outputs.data, 1)
            
            # Calculate number of correct and incorrect predictions
            correct_predictions += (predicted == labels).sum().item()
            incorrect_predictions += (predicted != labels).sum().item()
            total_predictions += labels.size(0)

    # Calculate accuracy
    accuracy = 100 * correct_predictions / total_predictions
    
    # Calculate Symbol Error Rate (SER)
    ser = incorrect_predictions / total_predictions

    # Calculate average loss
    average_loss = total_loss / len(test_loader)

    logger.info(f'Validation/Test Loss: {average_loss:.4f}')
    logger.info(f'Validation/Test Accuracy: {accuracy:.2f}%')
    logger.info(f'Symbol Error Rate (SER): {ser:.6f}')
    return ser

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S', filename="load_model.log")
    logger = logging.getLogger(__name__)
    
    desc = "This program loads a trained CNN model.\nIt requires the path to the folder containing the model file and path to the folder containing test images."
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("-i", "--img_dir", help="Path to the directory containing test images", type=str, required=True)
    parser.add_argument("-m", "--model_path", help="Path to the trained model file. This should only point to the folder where the models are contained, and not the actual model file. Defaults to cnn_output folder created by the cnn_v1.py script", type=str, required=True, default="./cnn_output/")
    parser.add_argument("-c", "--config", help="Path to the config file", type=str, required=True)
    args = parser.parse_args()
    
    # load snr values from config.json
    configfile = args.config
    with open('configfile') as f:
        config = json.load(f)
    snr_values = config["snr_values"]
    
    # setup paths
    model_path = args.model_path
    img_dir = args.img_dir

    
    # prepare GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # initialize list for symbol error rates
    symbol_error_rates = []
    
    for snr in snr_values:
        logger.info(f"Calculating SER for snr condition: {snr}")
        
        # load model for the corresponding snr value
        model = load_model(os.path.join(model_path,f"model_{snr}_snr"), M=128) # TODO could change to include M in config.json
        model.to(device)
        
        dataset = CustomImageDataset(img_dir=img_dir, specific_label=snr, transform=transform) # format dataset
        test_loader = DataLoader(dataset, batch_size=32, shuffle=False) # prepare dataloader to parralelize data loading
        criterion = nn.CrossEntropyLoss() # loss function
        
        # calculate SER
        ser = evaluate_and_calculate_ser(model, test_loader, criterion)
        
        # Store the calculated SER for later plotting
        symbol_error_rates.append(ser)
        
    logger.info("All SER values have been calculated.")
    # Create the final plot of all SER values against specific values (e.g., SNR levels)
    plt.figure(figsize=(10, 6))
    plt.plot(snr_values, symbol_error_rates, marker='o', linestyle='-', color='b')
    plt.xlabel('SNR')
    plt.ylabel('Symbol Error Rate (SER)')
    plt.yscale('log')
    plt.ylim(1e-5, 1e0)
    plt.title('SNR vs Symbol Error Rate')
    plt.grid(True)

    # Save the final plot to the folder
    final_plot_filename = os.path.join(model_path, 'snr_vs_ser_final_plot.png') # TODO create output folder and change here
    plt.savefig(final_plot_filename)  # Save the final plot

    # Optional: Display the plot (if needed)
    #plt.show()

    # Close the plot to free memory
    plt.close()

    print(f"Final plot has been saved to {final_plot_filename}.")
    
    
    
    
    
    
    
    
        
    
    