import logging
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def validation_loss_and_ser(model, val_loader, criterion):
    """
    Calculates the SER and validation loss for the given model and validation dataset.
    Args:
        model (torch.nn.Module): The neural network model to be evaluated.
        val_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        criterion (torch.nn.Module): Loss function used to calculate the loss.
    Returns:
        ser (float): Symbol Error Rate, calculated as the ratio of incorrect predictions to total predictions.
        validation_loss (float): The average loss over the validation
    """
    model.eval()  # Set model to evaluation mode
    correct_predictions = 0
    total_predictions = 0
    total_loss = 0.0
    incorrect_predictions = 0
    
    with torch.no_grad(): # Disable gradient calculation
        # Iterate over the validation dataset
        for data in val_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1) # Get the predicted labels

            # Calculate prediction statistics
            correct_predictions += (predicted == labels).sum().item()
            incorrect_predictions += (predicted != labels).sum().item()
            total_predictions += labels.size(0)

    accuracy = 100 * correct_predictions / total_predictions
    ser = incorrect_predictions / total_predictions
    validation_loss = total_loss / len(val_loader)
    
    return ser, validation_loss

class EarlyStopper:
    """
    EarlyStopper is a utility class to help with early stopping during model training.
    Attributes:
        patience (int): Number of epochs with no improvement after which training will be stopped.
        min_delta (float): Minimum change in validation loss.
        counter (int): Counter for the number of epochs with no improvement.
        min_validation_loss (float): The minimum validation loss observed so far.
    """
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        """
        Checks if training should be stopped early based on the validation loss.
        Args:
            validation_loss (float): The current epoch's validation loss.
        Returns:
            bool: True if training should be stopped, False otherwise.
        """
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def train(model, train_loader, num_epochs, optimizer, criterion, val_loader, logger:logging.Logger, patience, min_delta):
    """
    Trains the given model using the provided training data loader, optimizer, and loss criterion.
    Args:
        model (torch.nn.Module): The neural network model to be trained.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        num_epochs (int): Number of epochs to train the model.
        optimizer (torch.optim.Optimizer): Optimizer for updating the model's parameters.
        criterion (torch.nn.Module): Loss function to be used for training.
        val_loader (torch.utils.data.DataLoader): DataLoader for the test dataset.
        logger (logging.Logger): Logger for logging training information.
    Returns:
        float: The final SER (Symbol Error Rate) after training.
    """
    for x in range(num_epochs):
        model.train()
        running_loss = 0.0

        # Iterate through batches of training data
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()  # Clear the gradients of the model's parameters
            outputs = model(inputs)  # Perform a forward pass
            loss = criterion(outputs, labels)  # Calculate the loss
            loss.backward()  # Compute the gradients (backpropagation)
            optimizer.step()  # Update the model's parameters
            running_loss += loss.item() # Add the loss to the running total

            # Log the loss every 100 steps and reset the running total
            if i % 100 == 99:
                logger.info(f'Epoch [{x+1}], Step [{i+1}], Loss: {running_loss / 100:.4f}')
                running_loss = 0.0

        # Evaluate and calculate SER after each epoch
        ser, validation_loss = validation_loss_and_ser(model=model, val_loader=val_loader, criterion=criterion)
        logger.info(f"SER for epoch {x}: {ser}, Validation loss: {validation_loss}")

        # Check if early stopping is triggered
        early_stopper = EarlyStopper(patience, min_delta)
        if early_stopper.early_stop(validation_loss):
            logger.info(f"Early stopping activated after epoch {x}")
            break
    
    return ser
            
    