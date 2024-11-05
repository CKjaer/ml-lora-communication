import logger
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model, train_loader, num_epochs, optimizer, criterion):
    model.train()
    for _ in range(num_epochs):
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
                # logger.info(f'Epoch [{epoch+1}], Step [{i+1}], Loss: {running_loss / 100:.4f}')
                running_loss = 0.0