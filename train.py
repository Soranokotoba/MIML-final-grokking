from datetime import datetime
from models import Transformer, TransformerModel, get_model
from load_yaml import load_config
from data_generation import get_dataset
from optimizers import get_optimizer
from lr_schedule import get_scheduler

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import numpy as np
from tqdm import tqdm

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def set_seed(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    np.random.seed(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_model(
        epochs: int,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: torch.utils.data.DataLoader,
        device: torch.device,
        criterion: nn.Module,
        scheduler: torch.optim.lr_scheduler.LambdaLR,
        eval_loader: torch.utils.data.DataLoader
):
    train_acc = []
    eval_acc = []

    for epoch in tqdm(range(epochs)):
        model.train()
        running_loss = 0.0
        correct_preds = 0
        total_samples = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            # foward pass
            inputs = inputs.to(device)
            outputs = model(inputs)
            labels = labels.to(device)
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

        # update the learning rate
        scheduler.step()

        # evaluation after every epoch
        model.eval()
        eval_loss = 0.0
        correct_preds = 0
        total_eval_samples = 0

        with torch.no_grad(): 
            for inputs, labels in eval_loader:
                inputs, labels = inputs.to(device), labels.to(device)
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

        if not (epoch + 1) % 100:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")
            print(f"Evaluation - Loss: {eval_loss:.4f}, Accuracy: {eval_accuracy:.4f}")
        
        # early stopping
        if epoch > 2000 and eval_acc[-2000] > 0.999:
            break
    return train_acc, eval_acc


def main(config_path: str, output_path: str, result_npz_path: str):
    # set random seed
    set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # read config file
    # config_path = "./config_base.yaml"
    config, config_str = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = get_model(config)
    model = model.to(device)
    try:
        model.dropout = nn.Dropout(config.train.dropout)
    except:
        pass
    
    optimizer = get_optimizer(model, config.optim)
    scheduler = get_scheduler(optimizer, config.optim.lr)
    criterion = nn.CrossEntropyLoss()
    epochs = config.train.epochs

    train_loader, eval_loader = get_dataset(device=device, **config.train)

    train_acc, eval_acc = train_model(
        epochs,
        model,
        optimizer,
        train_loader,
        device,
        criterion,
        scheduler,
        eval_loader
    )
    # np.savez('result.npz', train_acc=train_acc, eval_acc=eval_acc)
    np.savez(result_npz_path, train_acc=train_acc, eval_acc=eval_acc)
    
    plt.figure()
    t_step = torch.arange(1, epochs+1)
    plt.plot(t_step, train_acc, label="train")
    plt.plot(t_step, eval_acc, label="evaluation")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy(%)")
    plt.xscale('log')
    plt.legend()
    # plt.savefig("Accuracy.png")
    plt.savefig(output_path + ".png")

if __name__ == "__main__":
    config_path = "./config_base.yaml"
    output_path = "Accuracy.png"
    main(config_path, output_path, "result.npz")