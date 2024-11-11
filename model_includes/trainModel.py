import logging
import torch
import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(script_dir, '..')))
from model_includes.evalModel import evaluate_and_calculate_ser

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model, train_loader, num_epochs, optimizer, criterion, test_loader, logger:logging.Logger):
    for x in range(num_epochs):
        model.train()
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
        logger.info(f"SER for epoch {x}")
        ser=evaluate_and_calculate_ser(model=model, test_loader=test_loader, criterion=criterion)
    return ser
            
    