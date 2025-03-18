"""
Training functions for skin cancer classification models.

This module contains functions for training, saving, and monitoring 
models during the training process.
"""

import os
import time
import copy
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

import sys
sys.path.append('..')
from config import (
    MODEL_PATH, LEARNING_RATE, SCHEDULER_STEP_SIZE, 
    SCHEDULER_GAMMA, EARLY_STOPPING_PATIENCE, WEIGHT_DECAY,
    RANDOM_SEED, RESULTS_DIR, NUM_EPOCHS
)


class EarlyStopping:
    """
    Early stopping to prevent overfitting.
    
    This class implements early stopping based on validation loss
    to prevent overfitting during training.
    """
    
    def __init__(self, patience=EARLY_STOPPING_PATIENCE, verbose=True, delta=0, path=MODEL_PATH):
        """
        Initialize EarlyStopping.
        
        Args:
            patience (int): How many epochs to wait after validation loss stopped improving
            verbose (bool): If True, prints a message for each validation loss improvement
            delta (float): Minimum change in the monitored quantity to qualify as an improvement
            path (str): Path to save the model checkpoint
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta
        self.path = path
    
    def __call__(self, val_loss, model):
        """
        Check if training should be stopped.
        
        Args:
            val_loss (float): Validation loss
            model (torch.nn.Module): Model to save if validation loss improves
            
        Returns:
            bool: True if model improved, False otherwise
        """
        score = -val_loss
        
        if self.best_score is None:
            # First epoch
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            return True
        
        elif score < self.best_score + self.delta:
            # Validation loss increased or didn't improve enough
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
            return False
            
        else:
            # Validation loss improved
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
            return True
    
    def save_checkpoint(self, val_loss, model):
        """
        Save model when validation loss decreases.
        
        Args:
            val_loss (float): Validation loss
            model (torch.nn.Module): Model to save
        """
        if self.verbose:
            improvement = self.val_loss_min - val_loss
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}) | Improvement: {improvement:.6f}')
        
        # Save model
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def train_model(model, train_loader, val_loader, criterion=None, optimizer=None,
                scheduler=None, num_epochs=NUM_EPOCHS, device=None):
    """
    Train a model with the given data loaders.
    
    Args:
        model (torch.nn.Module): Model to train
        train_loader (torch.utils.data.DataLoader): Training data loader
        val_loader (torch.utils.data.DataLoader): Validation data loader
        criterion (torch.nn.Module, optional): Loss function
        optimizer (torch.optim.Optimizer, optional): Optimizer
        scheduler (torch.optim.lr_scheduler._LRScheduler, optional): Learning rate scheduler
        num_epochs (int): Number of epochs to train for
        device (torch.device, optional): Device to train on
        
    Returns:
        tuple: (best_model_state_dict, history)
    """
    # Set random seed for reproducibility
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    # Use GPU if available
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    # Move model to device
    model = model.to(device)
    
    # Default criterion if not provided
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    
    # Default optimizer if not provided
    if optimizer is None:
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY
        )
    
    # Default scheduler if not provided
    if scheduler is None:
        scheduler = lr_scheduler.StepLR(
            optimizer, 
            step_size=SCHEDULER_STEP_SIZE,
            gamma=SCHEDULER_GAMMA
        )
    
    # Initialize early stopping
    early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE, verbose=True)
    
    # Initialize variables to track best model and training history
    best_model_state_dict = copy.deepcopy(model.state_dict())
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'lr': []
    }
    
    # Initialize timers
    since = time.time()
    
    # Main training loop
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
                dataloader = train_loader
            else:
                model.eval()   # Set model to evaluate mode
                dataloader = val_loader
            
            running_loss = 0.0
            running_corrects = 0
            
            # Iterate over data
            # Using tqdm for progress bar
            pbar = tqdm(dataloader, desc=phase)
            for inputs, labels in pbar:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass - track history only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                # Update progress bar
                pbar.set_postfix({'loss': loss.item()})
            
            # Calculate epoch metrics
            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)
            
            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # Store history
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
                history['lr'].append(optimizer.param_groups[0]['lr'])
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())
                
                # Check if early stopping
                improved = early_stopping(epoch_loss, model)
                if improved:
                    best_model_state_dict = copy.deepcopy(model.state_dict())
                
                if early_stopping.early_stop:
                    print("Early stopping triggered!")
                    break
        
        # Step the scheduler after validation
        if scheduler is not None:
            scheduler.step()
            
        print()
        
        # Break if early stopping
        if early_stopping.early_stop:
            break
    
    # Calculate training time
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    
    # Print best validation accuracy
    best_epoch = np.argmin(history['val_loss'])
    print(f'Best validation loss: {history["val_loss"][best_epoch]:.4f}')
    print(f'Corresponding validation accuracy: {history["val_acc"][best_epoch]:.4f}')
    
    # Plot training history
    plot_training_history(history)
    
    # Load best model weights
    model.load_state_dict(best_model_state_dict)
    
    return model, history


def plot_training_history(history, save_path=None):
    """
    Plot the training history.
    
    Args:
        history (dict): Dictionary containing training history
        save_path (str, optional): Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    ax1.plot(history['train_loss'], label='Train')
    ax1.plot(history['val_loss'], label='Validation')
    ax1.set_title('Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracy
    ax2.plot(history['train_acc'], label='Train')
    ax2.plot(history['val_acc'], label='Validation')
    ax2.set_title('Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    # Show the learning rate changes
    for i in range(1, len(history['lr'])):
        if history['lr'][i] != history['lr'][i-1]:
            ax1.axvline(x=i, color='r', linestyle='--', alpha=0.3)
            ax2.axvline(x=i, color='r', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
    
    plt.show()


def save_training_history(history, filename=None):
    """
    Save training history to a CSV file.
    
    Args:
        history (dict): Dictionary containing training history
        filename (str, optional): File path to save the history
    """
    if filename is None:
        filename = os.path.join(RESULTS_DIR, 'training_history.csv')
    
    # Convert to DataFrame
    df = pd.DataFrame(history)
    
    # Save to CSV
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    df.to_csv(filename, index=False)
    
    print(f"Training history saved to {filename}")


def fine_tune_model(model, train_loader, val_loader, layers_to_unfreeze=None, 
                   learning_rate=LEARNING_RATE/10, num_epochs=5, device=None):
    """
    Fine-tune a pre-trained model by unfreezing specific layers.
    
    Args:
        model (torch.nn.Module): Pre-trained model to fine-tune
        train_loader (torch.utils.data.DataLoader): Training data loader
        val_loader (torch.utils.data.DataLoader): Validation data loader
        layers_to_unfreeze (list, optional): List of layer indices to unfreeze
            If None, unfreeze all layers
        learning_rate (float): Learning rate for fine-tuning
        num_epochs (int): Number of epochs for fine-tuning
        device (torch.device, optional): Device to train on
        
    Returns:
        tuple: (fine_tuned_model, history)
    """
    print("Starting fine-tuning...")
    
    # Unfreeze specified layers
    if layers_to_unfreeze is None:
        # Unfreeze all layers
        model.unfreeze_base_model()
        print("Unfreezing all layers")
    else:
        # Unfreeze specific layers
        model.unfreeze_base_model(layers_to_unfreeze)
        print(f"Unfreezing layers: {layers_to_unfreeze}")
    
    # Fine-tuning optimizer with lower learning rate
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=WEIGHT_DECAY
    )
    
    # Learning rate scheduler
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )
    
    # Use different path for fine-tuned model
    model_path = str(MODEL_PATH).replace('.pth', '_fine_tuned.pth')
    
    # Custom early stopping for fine-tuning
    early_stopping = EarlyStopping(
        patience=EARLY_STOPPING_PATIENCE,
        verbose=True,
        path=model_path
    )
    
    # Training criterion
    criterion = nn.CrossEntropyLoss()
    
    # Fine-tune the model
    fine_tuned_model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=None,  # We'll handle the scheduler manually
        num_epochs=num_epochs,
        device=device
    )
    
    print("Fine-tuning complete!")
    
    return fine_tuned_model, history