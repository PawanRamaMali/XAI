"""
Evaluation functions for skin cancer classification models.

This module contains functions for evaluating model performance,
including metrics calculation and visualization.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, 
    auc, precision_recall_curve, average_precision_score
)
from tqdm import tqdm

import torch
import torch.nn.functional as F

import sys
sys.path.append('..')
from config import RESULTS_DIR, DPI


def evaluate_model(model, data_loader, device=None, return_predictions=False):
    """
    Evaluate a model on a dataset.
    
    Args:
        model (torch.nn.Module): Model to evaluate
        data_loader (torch.utils.data.DataLoader): Data loader for evaluation
        device (torch.device, optional): Device to evaluate on
        return_predictions (bool): Whether to return predictions
        
    Returns:
        dict: Dictionary containing evaluation metrics
        (and predictions if return_predictions=True)
    """
    # Use GPU if available
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Move model to device
    model = model.to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    # Initialize lists to store predictions and ground truth
    all_logits = []
    all_probs = []
    all_preds = []
    all_labels = []
    
    # No gradient computation needed for evaluation
    with torch.no_grad():
        # Iterate over batches
        for inputs, labels in tqdm(data_loader, desc="Evaluating"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Forward pass
            logits = model(inputs)
            
            # Get probabilities
            probs = F.softmax(logits, dim=1)
            
            # Get predictions
            _, preds = torch.max(logits, 1)
            
            # Store batch results
            all_logits.append(logits.cpu())
            all_probs.append(probs.cpu())
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
    
    # Concatenate batch results
    all_logits = torch.cat(all_logits, dim=0).numpy()
    all_probs = torch.cat(all_probs, dim=0).numpy()
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    
    # Calculate accuracy
    correct = (all_preds == all_labels).sum()
    total = len(all_labels)
    accuracy = correct / total
    
    # Create metrics dictionary
    metrics = {
        'accuracy': accuracy,
        'num_samples': total,
        'num_correct': correct
    }
    
    # Add predictions if requested
    if return_predictions:
        metrics.update({
            'logits': all_logits,
            'probabilities': all_probs,
            'predictions': all_preds,
            'labels': all_labels
        })
    
    return metrics


def plot_confusion_matrix(labels, predictions, class_names, save_path=None):
    """
    Plot a confusion matrix.
    
    Args:
        labels (numpy.ndarray): Ground truth labels
        predictions (numpy.ndarray): Predicted labels
        class_names (list): List of class names
        save_path (str, optional): Path to save the plot
    """
    # Calculate confusion matrix
    cm = confusion_matrix(labels, predictions)
    
    # Normalize by row (true labels)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Plot using seaborn
    sns.heatmap(
        cm_norm, 
        annot=True, 
        fmt='.2f', 
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    
    plt.title('Confusion Matrix (Normalized)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    
    plt.show()
    
    # Also plot raw counts
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    
    plt.title('Confusion Matrix (Counts)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_path:
        new_path = save_path.replace('.png', '_counts.png')
        plt.savefig(new_path, dpi=DPI, bbox_inches='tight')
    
    plt.show()


def plot_roc_curves(labels, probabilities, class_names, save_path=None):
    """
    Plot ROC curves for each class.
    
    Args:
        labels (numpy.ndarray): Ground truth labels (one-hot encoded)
        probabilities (numpy.ndarray): Class probabilities
        class_names (list): List of class names
        save_path (str, optional): Path to save the plot
    """
    # Convert labels to one-hot encoding if they're not already
    if len(labels.shape) == 1:
        num_classes = len(class_names)
        labels_one_hot = np.zeros((labels.size, num_classes))
        labels_one_hot[np.arange(labels.size), labels] = 1
    else:
        labels_one_hot = labels
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Plot ROC curve for each class
    for i, class_name in enumerate(class_names):
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(labels_one_hot[:, i], probabilities[:, i])
        roc_auc = auc(fpr, tpr)
        
        # Plot
        plt.plot(
            fpr, tpr, 
            lw=2, 
            label=f'{class_name} (AUC = {roc_auc:.2f})'
        )
    
    # Plot random chance line
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    
    # Set plot parameters
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    
    plt.show()


def plot_precision_recall_curves(labels, probabilities, class_names, save_path=None):
    """
    Plot precision-recall curves for each class.
    
    Args:
        labels (numpy.ndarray): Ground truth labels (one-hot encoded)
        probabilities (numpy.ndarray): Class probabilities
        class_names (list): List of class names
        save_path (str, optional): Path to save the plot
    """
    # Convert labels to one-hot encoding if they're not already
    if len(labels.shape) == 1:
        num_classes = len(class_names)
        labels_one_hot = np.zeros((labels.size, num_classes))
        labels_one_hot[np.arange(labels.size), labels] = 1
    else:
        labels_one_hot = labels
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Plot precision-recall curve for each class
    for i, class_name in enumerate(class_names):
        # Calculate precision-recall curve
        precision, recall, _ = precision_recall_curve(
            labels_one_hot[:, i], 
            probabilities[:, i]
        )
        
        # Calculate average precision
        ap = average_precision_score(labels_one_hot[:, i], probabilities[:, i])
        
        # Plot
        plt.plot(
            recall, precision, 
            lw=2, 
            label=f'{class_name} (AP = {ap:.2f})'
        )
    
    # Set plot parameters
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend(loc="best")
    plt.grid(True)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    
    plt.show()


def print_classification_report(labels, predictions, class_names):
    """
    Print a classification report.
    
    Args:
        labels (numpy.ndarray): Ground truth labels
        predictions (numpy.ndarray): Predicted labels
        class_names (list): List of class names
    """
    # Create classification report
    report = classification_report(
        labels, 
        predictions, 
        target_names=class_names,
        digits=3
    )
    
    print("Classification Report:")
    print(report)
    
    # Calculate per-class accuracy
    class_correct = {}
    class_total = {}
    
    for i in range(len(labels)):
        label = labels[i]
        pred = predictions[i]
        
        if label not in class_total:
            class_total[label] = 0
            class_correct[label] = 0
            
        class_total[label] += 1
        if label == pred:
            class_correct[label] += 1
    
    print("\nPer-Class Accuracy:")
    for i, class_name in enumerate(class_names):
        if i in class_total and class_total[i] > 0:
            accuracy = class_correct[i] / class_total[i]
            print(f"{class_name}: {accuracy:.3f} ({class_correct[i]}/{class_total[i]})")


def plot_misclassified_examples(model, data_loader, class_names, device=None, 
                               num_examples=10, save_path=None):
    """
    Plot examples of misclassified images.
    
    Args:
        model (torch.nn.Module): Model to evaluate
        data_loader (torch.utils.data.DataLoader): Data loader
        class_names (list): List of class names
        device (torch.device, optional): Device to evaluate on
        num_examples (int): Number of examples to plot
        save_path (str, optional): Path to save the plot
    """
    # Use GPU if available
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Move model to device
    model = model.to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    # Lists to store misclassified examples
    misclassified_images = []
    misclassified_labels = []
    misclassified_preds = []
    misclassified_probs = []
    
    # No gradient computation needed for evaluation
    with torch.no_grad():
        for inputs, labels in data_loader:
            # Skip if we already have enough examples
            if len(misclassified_images) >= num_examples:
                break
                
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Forward pass
            logits = model(inputs)
            
            # Get probabilities
            probs = F.softmax(logits, dim=1)
            
            # Get predictions
            _, preds = torch.max(logits, 1)
            
            # Find misclassified examples in this batch
            misclassified_idx = (preds != labels).nonzero(as_tuple=True)[0]
            
            for idx in misclassified_idx:
                # Skip if we already have enough examples
                if len(misclassified_images) >= num_examples:
                    break
                    
                # Store misclassified example
                misclassified_images.append(inputs[idx].cpu())
                misclassified_labels.append(labels[idx].item())
                misclassified_preds.append(preds[idx].item())
                misclassified_probs.append(probs[idx].cpu().numpy())
    
    # Check if we found any misclassified examples
    if len(misclassified_images) == 0:
        print("No misclassified examples found.")
        return
    
    # Plot misclassified examples
    num_cols = 5
    num_rows = (len(misclassified_images) + num_cols - 1) // num_cols
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 3 * num_rows))
    axes = axes.flatten() if num_rows > 1 else [axes] if num_cols == 1 else axes
    
    for i, (img, label, pred, prob) in enumerate(zip(
        misclassified_images, misclassified_labels, 
        misclassified_preds, misclassified_probs)):
        
        # Convert to numpy and denormalize
        img = img.permute(1, 2, 0).numpy()
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)
        
        # Plot image
        axes[i].imshow(img)
        axes[i].set_title(
            f"True: {class_names[label]}\nPred: {class_names[pred]}\n"
            f"Confidence: {prob[pred]:.2f}"
        )
        axes[i].axis('off')
    
    # Hide empty subplots
    for i in range(len(misclassified_images), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    
    plt.show()


def evaluate_and_save_results(model, data_loader, class_names, save_dir=RESULTS_DIR, device=None):
    """
    Evaluate a model and save various performance metrics and visualizations.
    
    Args:
        model (torch.nn.Module): Model to evaluate
        data_loader (torch.utils.data.DataLoader): Data loader for evaluation
        class_names (list): List of class names
        save_dir (str): Directory to save results
        device (torch.device, optional): Device to evaluate on
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    print("Evaluating model...")
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Evaluate model
    metrics = evaluate_model(model, data_loader, device, return_predictions=True)
    
    # Extract predictions and labels
    predictions = metrics['predictions']
    labels = metrics['labels']
    probabilities = metrics['probabilities']
    
    # Print overall accuracy
    accuracy = metrics['accuracy']
    print(f"Overall accuracy: {accuracy:.4f} ({metrics['num_correct']}/{metrics['num_samples']})")
    
    # Save metrics to CSV
    df = pd.DataFrame({
        'accuracy': accuracy,
        'num_samples': metrics['num_samples'],
        'num_correct': metrics['num_correct']
    }, index=[0])
    
    df.to_csv(os.path.join(save_dir, 'metrics.csv'), index=False)
    
    # Print detailed classification report
    print_classification_report(labels, predictions, class_names)
    
    # Plot confusion matrix
    plot_confusion_matrix(
        labels, predictions, class_names,
        save_path=os.path.join(save_dir, 'confusion_matrix.png')
    )
    
    # Plot ROC curves
    plot_roc_curves(
        labels, probabilities, class_names,
        save_path=os.path.join(save_dir, 'roc_curves.png')
    )
    
    # Plot precision-recall curves
    plot_precision_recall_curves(
        labels, probabilities, class_names,
        save_path=os.path.join(save_dir, 'precision_recall_curves.png')
    )
    
    # Plot misclassified examples
    plot_misclassified_examples(
        model, data_loader, class_names, device,
        num_examples=10,
        save_path=os.path.join(save_dir, 'misclassified_examples.png')
    )
    
    print(f"Evaluation results saved to {save_dir}")
    
    return metrics