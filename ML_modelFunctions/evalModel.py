import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    print(f'Validation/Test Loss: {average_loss:.4f}')
    print(f'Validation/Test Accuracy: {accuracy:.2f}%')
    print(f'Symbol Error Rate (SER): {ser:.6f}')
    # logger.info(f'Validation/Test Loss: {average_loss:.4f}')
    # logger.info(f'Validation/Test Accuracy: {accuracy:.2f}%')
    # logger.info(f'Symbol Error Rate (SER): {ser:.6f}')
    return ser, average_loss