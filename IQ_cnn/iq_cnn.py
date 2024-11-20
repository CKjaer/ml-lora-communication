import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchsummary import summary
import logging
from tqdm import tqdm
from iq_dataset import IQDataset, CustomIQTransform
from torch.utils.tensorboard import SummaryWriter

class IQCNN(nn.Module):
    def __init__(self, M): # M is number of symbols
        super(IQCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=M//4, kernel_size=3, stride=1, padding=1), # specify dtype for complex numbers
            nn.BatchNorm1d(M//4),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=(2), stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=M//4, out_channels=M//2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(M//2),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=2, stride=2),
            nn.Flatten()
        )
        
        self.fc1 = nn.Sequential(
            nn.Linear(32*(M//2), 4 * M), # M//2 channels of 32 features each, to 4M features
            nn.BatchNorm1d(4 * M),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(4 * M, 2 * M),
            nn.BatchNorm1d(2 * M),
            nn.ReLU()
        )
        self.output_layer = nn.Sequential(
            nn.Linear(2 * M, M)
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x) # flatten is inside conv2
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.output_layer(x)
        return x

class otherArticleIQCNN(nn.Module):
    def __init__(self, M): # M is number of symbols
        super(otherArticleIQCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=64, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=1)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=1),
            nn.Flatten()
        )
        self.output_layer = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(256*125, M)
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.output_layer(x)
        return x

def train(model: nn.Module, train_loader: DataLoader, evaluation_loader: DataLoader, num_epochs: int, criterion: nn.Module, optimizer: nn.Module, device, logger: logging.Logger, writer: SummaryWriter):
    """
    Loop function to train the model for given amount of epochs
    Function evaluates the model after each epoch and logs the results

    Args:
        model (nn.Module): The model to train
        train_loader (DataLoader): Data loader feeding batches of training data
        evaluation_loader (DataLoader): Data loader feeding batches of evaluation data
        num_epochs (int): Number of epochs to train
        criterion (nn.Module): Loss function
        optimizer (nn.Module): Optimizer function
        device (_type_): GPU or CPU
        logger (logging.Logger): Logger object
        writer (SummaryWriter): TensorBoard writer object

    Returns:
        float: Symbol error rate after training. Note that this only returns the final SER, as the function is a loop function that runs until num_epochs.
    """
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}")
        for i, (data, labels) in progress_bar:
            data = data.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if i % 100 == 0:
                avg_loss = running_loss / 10
                progress_bar.set_postfix(loss=avg_loss)
                logger.info(f"Epoch: {epoch+1}, Step: {i+1}, Loss: {avg_loss:.4f}")
                writer.add_scalar('Training Loss', avg_loss, epoch * len(train_loader) + i)
                running_loss = 0.0
        
        # Evaluate the model after each epoch
        ser = evaluate_and_calculate_ser(model, evaluation_loader, criterion, device, logger, writer, epoch)
        logger.info(f"SER after epoch {epoch+1}: {ser:.6f}")
    
    # Return the SER of the final epoch
    return ser

# Evaluation function (called inside the training loop)
def evaluate_and_calculate_ser(model, evaluation_loader, criterion, device, logger, writer, epoch):
    model.eval()  # Set model to evaluation mode
    correct_predictions = 0
    total_predictions = 0
    total_loss = 0.0
    incorrect_predictions = 0

    with torch.no_grad():
        for data in evaluation_loader:
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
    average_loss = total_loss / len(evaluation_loader)

    logger.info(f'Validation/Test Loss: {average_loss:.4f}')
    logger.info(f'Validation/Test Accuracy: {accuracy:.2f}%')
    logger.info(f'Symbol Error Rate (SER): {ser:.6f}')
    
    # Log metrics to TensorBoard
    writer.add_scalar('Validation Loss', average_loss, epoch)
    writer.add_scalar('Validation Accuracy', accuracy, epoch)
    writer.add_scalar('Symbol Error Rate', ser, epoch)
    
    return ser

if __name__ == "__main__":
    # create a logger
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S', filename="IQ_model_debugging.log", encoding='utf-8', level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Check if GPU is available otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    logger.info(f"Using device: {device}")

    # Define the model
    M = 128
    model = otherArticleIQCNN(M).to(device)
    logger.info(summary(model, (2, 128)))

    dataset = IQDataset("output/20241114-115337/csv", transform=CustomIQTransform(), logger=logger)
    dataset.subset_data(snr=-6, rate_param=0.0)
    train_set, val_set = random_split(dataset, [int(0.8*len(dataset)), len(dataset) - int(0.8*len(dataset))]) # 80-20 split
    
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=True)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    writer = SummaryWriter()

    train(model, train_loader, val_loader, 10, criterion, optimizer, device, logger, writer)