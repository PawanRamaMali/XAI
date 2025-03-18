"""
General plotting functions for skin cancer classification and explainability.

This module provides utility functions for creating visualizations
of model predictions, explanations, and performance metrics.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.manifold import TSNE

import torch
import torch.nn.functional as F

import sys
sys.path.append('..')
from config import DPI, COLORMAP, OVERLAY_ALPHA


def plot_batch_samples(images, labels, class_names, n_samples=16, figsize=(12, 12), save_path=None):
    """
    Plot a grid of sample images from a batch.
    
    Args:
        images (torch.Tensor): Batch of images (B, C, H, W)
        labels (torch.Tensor): Batch of labels (B)
        class_names (list): List of class names
        n_samples (int): Number of samples to plot
        figsize (tuple): Figure size
        save_path (str, optional): Path to save the plot
    """
    # Determine number of rows and cols
    n_cols = int(np.sqrt(n_samples))
    n_rows = int(np.ceil(n_samples / n_cols))
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]
    
    # Plot each sample
    for i in range(min(n_samples, len(images))):
        # Get image and label
        img = images[i].permute(1, 2, 0).cpu().numpy()
        label = labels[i].item()
        
        # Denormalize
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)
        
        # Plot
        axes[i].imshow(img)
        axes[i].set_title(f"{class_names[label]}")
        axes[i].axis('off')
    
    # Hide empty subplots
    for i in range(len(images), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    
    plt.show()


def plot_heatmap_overlay(image, heatmap, alpha=OVERLAY_ALPHA, colormap=COLORMAP,
                        figsize=(12, 4), title=None, save_path=None):
    """
    Plot an image with a heatmap overlay.
    
    Args:
        image (numpy.ndarray): Original image array of shape (H, W, 3)
        heatmap (numpy.ndarray): Heatmap array of shape (H, W)
        alpha (float): Transparency of the heatmap overlay
        colormap (str): Matplotlib colormap to use for the heatmap
        figsize (tuple): Figure size
        title (str, optional): Plot title
        save_path (str, optional): Path to save the plot
    """
    # Create figure
    plt.figure(figsize=figsize)
    
    # Get colormap
    cmap = plt.get_cmap(colormap)
    
    # Normalize heatmap
    heatmap_norm = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-8)
    
    # Create heatmap image
    heatmap_rgb = cmap(heatmap_norm)[:, :, :3]
    
    # Create overlay
    overlay = (1 - alpha) * image + alpha * heatmap_rgb
    overlay = np.clip(overlay, 0, 1)
    
    # Create subplots
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(heatmap_rgb)
    plt.title("Heatmap")
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(overlay)
    plt.title("Overlay")
    plt.axis('off')
    
    if title:
        plt.suptitle(title, fontsize=16)
        
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    
    plt.show()


def plot_feature_embedding(model, data_loader, class_names, layer_name=None,
                         n_samples=1000, device=None, figsize=(10, 8), save_path=None):
    """
    Plot t-SNE embedding of features from a specific layer.
    
    Args:
        model (torch.nn.Module): The model
        data_loader (torch.utils.data.DataLoader): Data loader
        class_names (list): List of class names
        layer_name (str, optional): Name of the layer to extract features from
            If None, uses the features before the classifier
        n_samples (int): Number of samples to use
        device (torch.device, optional): Device to run the model on
        figsize (tuple): Figure size
        save_path (str, optional): Path to save the plot
    """
    # Set device
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Move model to device
    model = model.to(device)
    model.eval()
    
    # Create a hook to get the features
    features = []
    labels = []
    
    def hook_fn(module, input, output):
        features.append(output.detach().cpu())
    
    # Register the hook
    if layer_name is None:
        # By default, get features from the base model
        if hasattr(model, 'base_model'):
            hook = model.base_model.register_forward_hook(hook_fn)
        else:
            # If no base_model, use the model itself
            hook = model.register_forward_hook(hook_fn)
    else:
        # Get the specified layer
        module = model
        for name in layer_name.split('.'):
            module = getattr(module, name)
        hook = module.register_forward_hook(hook_fn)
    
    # Collect features
    with torch.no_grad():
        for inputs, targets in data_loader:
            if len(features) * inputs.size(0) >= n_samples:
                # Collect up to n_samples
                inputs = inputs[:n_samples - len(features) * inputs.size(0)]
                targets = targets[:n_samples - len(features) * inputs.size(0)]
                
            if len(inputs) == 0:
                break
                
            # Forward pass
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            # Store labels
            labels.append(targets)
            
            if len(features) * inputs.size(0) >= n_samples:
                break
    
    # Remove the hook
    hook.remove()
    
    # Concatenate features and labels
    features = torch.cat(features, dim=0).reshape(len(labels) * labels[0].size(0), -1)
    labels = torch.cat(labels, dim=0)
    
    # Only keep n_samples
    features = features[:n_samples]
    labels = labels[:n_samples]
    
    # Convert to numpy
    features = features.numpy()
    labels = labels.numpy()
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    features_tsne = tsne.fit_transform(features)
    
    # Plot
    plt.figure(figsize=figsize)
    
    for i, class_name in enumerate(class_names):
        # Get indices of samples with this class
        idx = labels == i
        if idx.sum() > 0:
            plt.scatter(
                features_tsne[idx, 0], features_tsne[idx, 1],
                label=class_name, alpha=0.7
            )
    
    plt.legend()
    plt.title('t-SNE of Features')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    
    plt.show()


def plot_image_grid(images, titles=None, figsize=None, rows=None, cols=None, save_path=None):
    """
    Plot a grid of images.
    
    Args:
        images (list): List of images to plot
        titles (list, optional): List of titles for each image
        figsize (tuple, optional): Figure size
        rows (int, optional): Number of rows
        cols (int, optional): Number of columns
        save_path (str, optional): Path to save the plot
    """
    # Determine number of rows and cols
    n = len(images)
    if rows is None and cols is None:
        cols = int(np.sqrt(n))
        rows = int(np.ceil(n / cols))
    elif rows is None:
        rows = int(np.ceil(n / cols))
    elif cols is None:
        cols = int(np.ceil(n / rows))
    
    # Determine figure size
    if figsize is None:
        figsize = (cols * 4, rows * 4)
    
    # Create figure
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    
    # Convert to 2D array of axes
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = np.array([axes])
    elif cols == 1:
        axes = np.array([[ax] for ax in axes])
    
    # Plot each image
    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            if idx < n:
                axes[i, j].imshow(images[idx])
                if titles is not None and idx < len(titles):
                    axes[i, j].set_title(titles[idx])
                axes[i, j].axis('off')
            else:
                axes[i, j].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    
    plt.show()


def plot_attention_maps(image, attention_maps, titles=None, figsize=None, save_path=None):
    """
    Plot attention maps from a model.
    
    Args:
        image (numpy.ndarray): Original image array of shape (H, W, 3)
        attention_maps (list): List of attention maps, each of shape (H, W)
        titles (list, optional): List of titles for each attention map
        figsize (tuple, optional): Figure size
        save_path (str, optional): Path to save the plot
    """
    # Number of attention maps
    n = len(attention_maps)
    
    # Default titles
    if titles is None:
        titles = [f"Attention Map {i+1}" for i in range(n)]
    
    # Determine figure size
    if figsize is None:
        figsize = (4 * (n + 1), 4)
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Plot original image
    plt.subplot(1, n + 1, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis('off')
    
    # Plot attention maps
    for i in range(n):
        plt.subplot(1, n + 1, i + 2)
        plt.imshow(attention_maps[i], cmap='viridis')
        plt.title(titles[i])
        plt.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    
    plt.show()


def plot_prediction_confidence(probabilities, class_names, figsize=(10, 6), save_path=None):
    """
    Plot prediction confidence for all classes.
    
    Args:
        probabilities (numpy.ndarray): Class probabilities
        class_names (list): List of class names
        figsize (tuple): Figure size
        save_path (str, optional): Path to save the plot
    """
    # Create figure
    plt.figure(figsize=figsize)
    
    # Create horizontal barplot
    y_pos = np.arange(len(class_names))
    plt.barh(y_pos, probabilities)
    
    # Add labels
    plt.yticks(y_pos, class_names)
    plt.xlabel('Confidence')
    plt.title('Prediction Confidence')
    
    # Add grid
    plt.grid(True, axis='x')
    
    # Add values
    for i, prob in enumerate(probabilities):
        plt.text(prob + 0.01, i, f"{prob:.2f}", va='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    
    plt.show()