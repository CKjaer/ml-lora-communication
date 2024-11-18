import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchsummary import summary
import logging
from iq_dataset import IQDataset, CustomIQTransform

class IQCNN(nn.Module):
    def __init__(self, M): # M is number of symbols
        super(IQCNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=M//4, kernel_size=(1,3), stride=1, padding=(0,1)), # specify dtype for complex numbers
            nn.BatchNorm2d(M//4),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1,2), stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=M//4, out_channels=M//2, kernel_size=(1,3), stride=1, padding=(0,1)),
            nn.BatchNorm2d(M//2),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1,2), stride=2),
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
            nn.Linear(2 * M, 128),
        )
        
    def forward(self, x):
        print(x.shape)
        x = x.unsqueeze(2) # add channel dimension
        print(x.shape)
        x = self.conv1(x)
        x = self.conv2(x) # flatten is inside conv2
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.output_layer(x)
        return x

def train(model, train_loader, evaluation_loader, num_epochs, criterion, optimizer, device, logger):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (data, labels) in enumerate(train_loader):
            data = data.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if i % 100 == 99:
                avg_loss = running_loss / 100
                print(f"Epoch: {epoch}, Step: {i+1}, Loss: {avg_loss:.4f}")
                logger.info(f"Epoch: {epoch}, Step: {i+1}, Loss: {avg_loss:.4f}")
                running_loss = 0.0
        
        logger.info(f"Evaluation after training epoch: {epoch}")
        ser = evaluate_and_calculate_ser(model, evaluation_loader, criterion, device, logger)
        logger.info(f"SER after epoch {epoch}: {ser:.6f}")
    return ser

# Evaluation function (called inside the training loop)
def evaluate_and_calculate_ser(model, evaluation_loader, criterion, device, logger):
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
    return ser

if __name__ == "__main__":
    # create a logger
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S', filename="train.log", encoding='utf-8', level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Check if GPU is available otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define the model
    M = 128
    model = IQCNN(M).to(device)

    dataset = IQDataset("output/20241114-115337/csv", snr=-6, rate_param=0.0, transform=CustomIQTransform(), logger=logger)
    train_set, val_set = random_split(dataset, [int(0.8*len(dataset)), len(dataset) - int(0.8*len(dataset))]) # 80-20 split
    
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=True)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    
    train(model, train_loader, val_loader, 10, criterion, optimizer, device, logger)

    
    
    