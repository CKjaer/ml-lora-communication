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

# Configure logger
logfilename = "cnn.log"
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s',datefmt='%Y-%m-%d %H:%M:%S', filename=logfilename, encoding='utf-8', level=logging.INFO)
logger.info("Starting the program")




# Define the CNN architecture
class LoRaCNN(nn.Module):
    def __init__(self, M):
        super(LoRaCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=M//4, kernel_size=4, stride=1, padding=2)
        self.conv2 = nn.Conv2d(in_channels=M//4, out_channels=M//2, kernel_size=4, stride=1, padding=2)
        
        # Pooling layer
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(M//2 * (M//4) * (M//4), 4 * M)  
        self.fc2 = nn.Linear(4 * M, 2 * M)
        self.fc3 = nn.Linear(2 * M, M)
        
    def forward(self, x):
        try:
            # Convolutional layers
            x = F.relu(self.conv1(x))  
            x = self.pool(x)           
            x = F.relu(self.conv2(x))  
            x = self.pool(x)           
            
            # Flatten the output from conv layers
            x = x.view(-1, self.num_flat_features(x))
            
            # Fully connected layers
            x = F.relu(self.fc1(x))    
            x = F.relu(self.fc2(x))    
            x = self.fc3(x)            
            
            return x
        except Exception as e:
            logger.error(f"Error in forward pass:{e}")
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # All dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


# Example training loop with GPU support
def train(model, train_loader, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        logger.info(f"Starting epoch {epoch + 1}/{num_epochs}")
        for i, data in enumerate(train_loader, 0):
            try:

                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU if available
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
            except Exception as e:
                logger.info(f"Error during training at epoch {epoch + 1}, batch {i + 1}: {e}")
            
            # Print statistics
            running_loss += loss.item()
            if i % 100 == 99:  # Print every 100 mini-batches
                logger.info(f'Epoch [{epoch+1}], Step [{i+1}], Loss: {running_loss / 100:.4f}')
                running_loss = 0.0


# Dataset class with error handling for images
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
        global corrupt_imag_count
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
            logger.error(f"Error loading image {img_name}: at index {idx} {e}")
            # If an error occurs, we can either return None or handle it as appropriate.
            # Corrupt image counter +1
            corrupt_imag_count += 1 
            return None, None

# Define transformations for the images
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize images to a fixed size
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize with mean and std
])



# Example evaluation function with GPU support
def evaluate_and_calculate_ser(model, test_loader, criterion):
    model.eval()  # Set model to evaluation mode
    correct_predictions = 0
    total_predictions = 0
    total_loss = 0.0
    incorrect_predictions = 0

    with torch.no_grad():  # Disable gradient calculation for evaluation
        for data in test_loader:
            try:

                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU if available
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
            except Exception as e:
                logger.error(f"Error during evaluation: {e}")
            
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


# The rest of the code for loading datasets, training, evaluating, and plotting remains the same.


if __name__ == "__main__":
    # Measure time
    start_time = time.time()

    #Counter for corrupted images
    corrupt_imag_count = 0
    
    # Define the directory to save the final plot
    img_dir = "./first_data_set/plots"  # Update this path accordingly
    output_folder = './cnn_output/'

    # Create the directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Configure logger
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S', filename=os.path.join(output_folder, 'cnn_log.log'))
    logger = logging.getLogger(__name__)
    
    # List of specific values for which SER will be calculated
    specific_values = [i for i in range(-16, -2, 2)]

    # Placeholder to store symbol error rates for each specific value
    symbol_error_rates = []

    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Loop over each specific value
    for value in specific_values:
        logger.info(f"Calculating SER for specific value: {value}")
        # Set path to the folder containing the images


        # Create the dataset and DataLoader
        dataset = CustomImageDataset(img_dir=img_dir, specific_label=float(value), transform=transform)


        # Assuming dataset is your complete dataset
        dataset_size = len(dataset)
        train_size = int(0.8 * dataset_size)  # 80% for training
        test_size = dataset_size - train_size  # 20% for testing

        # Split the dataset into training and test sets
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

        # Create DataLoaders for training and test sets
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        # Train the model using the train function defined previous
        

    # Check dataset and DataLoader
    logger.info(f"Number of images in dataset: {len(dataset)}")
    image, label = dataset[0]
    model = LoRaCNN(M).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

        train(model, train_loader, 3)
        
        torch.save(model.state_dict(), os.path.join(output_folder, f'model_{value}_snr.pth'))  # Save the model for future use
        torch.save(optimizer.state_dict(), os.path.join(output_folder,f'optimizer_{value}_snr.pth'))  # Save the optimizer for future use

        # Evaluate the model and calculate SER for the current specific value
        # Assuming that the specific value affects how the model is evaluated (e.g., different SNR levels)
        ser = evaluate_and_calculate_ser(model, test_loader, criterion)
        
        # Store the calculated SER for later plotting
        symbol_error_rates.append(ser)

    #Log the number of corrupted images
    logger.info(f"Number of corrupted images: {corrupt_imag_count}")
    
    logger.info("All SER values have been calculated.")

    # Create the final plot of all SER values against specific values (e.g., SNR levels)
    plt.figure(figsize=(10, 6))
    plt.plot(specific_values, symbol_error_rates, marker='o', linestyle='-', color='b')
    plt.xlabel('SNR')
    plt.ylabel('Symbol Error Rate (SER)')
    plt.yscale('log')
    plt.ylim(1e-5, 1e0)
    plt.title('SNR vs Symbol Error Rate')
    plt.grid(True)

    # Save the final plot to the folder
    final_plot_filename = os.path.join(output_folder, 'snr_vs_ser_final_plot.png')
    plt.savefig(final_plot_filename)  # Save the final plot

    # Optional: Display the plot (if needed)
    plt.show()

    # Close the plot to free memory
    plt.close()

    # Measure of the end time
    end_time = tiem.time()

    # Obtain the execution time
    execution_time = end_time - start_time
    logger.info(f"Execution time: {execution_time} seconds")

    print(f"Final plot has been saved to {final_plot_filename}.")
