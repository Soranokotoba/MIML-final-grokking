from models import get_model
from load_yaml import load_config
from data_generation import get_dataset

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os 

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def train_model():
    train_acc = []
    eval_acc = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_preds = 0
        total_samples = 0

        for inputs, labels in train_loader:
            optimizer.zero_grad()

            # foward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)  # caculate loss
            loss.backward()  # backward pass
            optimizer.step()  # update parameters
            
            running_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)

            # calculate accuracy
            _, predicted = torch.max(outputs, 1)
            correct_preds += (predicted == labels).sum().item()

        # print average training loss
        epoch_loss = running_loss / total_samples
        epoch_accuracy = correct_preds / total_samples
        train_acc.append(epoch_accuracy)

        # evaluation after every epoch
        model.eval()
        eval_loss = 0.0
        correct_preds = 0
        total_eval_samples = 0

        with torch.no_grad(): 
            for inputs, labels in eval_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                eval_loss += loss.item() * inputs.size(0)
                total_eval_samples += inputs.size(0)

                # calculate accuracy
                _, predicted = torch.max(outputs, 1)
                correct_preds += (predicted == labels).sum().item()

        # print evaluation results
        eval_loss = eval_loss / total_eval_samples
        eval_accuracy = correct_preds / total_eval_samples
        eval_acc.append(eval_accuracy)

        if not (epoch + 1) % 10:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")
            print(f"Evaluation - Loss: {eval_loss:.4f}, Accuracy: {eval_accuracy:.4f}")
    return train_acc, eval_acc


if __name__ == "__main__":
    config_path = "./config_base.yaml"
    config, config_str = load_config(config_path)

    model = get_model(config)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.98))
    criterion = nn.CrossEntropyLoss()
    epochs = config.train.epochs

    train_loader, eval_loader = get_dataset(**config.train)

    train_acc, eval_acc = train_model()
    plt.figure()
    t_step = torch.arange(1, epochs+1)
    plt.plot(t_step, train_acc, label="train")
    plt.plot(t_step, eval_acc, label="evaluation")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy(%)")
    plt.xscale('log')
    plt.legend()
    plt.savefig("Accuracy.png")
