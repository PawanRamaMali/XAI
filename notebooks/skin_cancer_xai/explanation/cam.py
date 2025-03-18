"""
Class Activation Mapping (CAM) implementation for skin cancer classification model.

This module implements the original CAM method for visualizing 
class-specific discriminative regions in an image.

Reference:
    Zhou et al., "Learning Deep Features for Discriminative Localization",
    https://arxiv.org/abs/1512.04150
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2

import torch
import torch.nn.functional as F

import sys
sys.path.append('..')
from config import COLORMAP, OVERLAY_ALPHA, DPI


class CAM:
    """
    Class Activation Mapping (CAM) for CNN models.
    
    This class implements the original CAM method, which requires a specific
    model architecture with a global average pooling layer followed by a
    fully connected layer.
    """
    
    def __init__(self, model, device=None):
        """
        Initialize CAM.
        
        Args:
            model (torch.nn.Module): The model to explain
            device (torch.device, optional): Device to run the model on
        """
        self.model = model
        self.model.eval()
        
        # Use GPU if available
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Find target layer (last convolutional layer) and fc weights
        self.target_layer, self.fc_weights = self._find_layers()
    
    def _find_layers(self):
        """
        Find the target convolutional layer and fc weights.
        
        Returns:
            tuple: (target_layer, fc_weights)
        """
        # This will depend on the specific model architecture
        # For ResNet models
        if 'resnet' in self.model.model_name:
            # Get the last convolutional layer
            target_layer = self.model.base_model.layer4[-1]
            
            # Get weights of the classifier
            fc_weights = self.model.classifier[-1].weight.data
            
        # For EfficientNet models
        elif 'efficientnet' in self.model.model_name:
            # Get the last convolutional layer
            target_layer = self.model.base_model.features[-1]
            
            # Get weights of the classifier
            fc_weights = self.model.classifier[-1].weight.data
            
        # For DenseNet models
        elif 'densenet' in self.model.model_name:
            # Get the last convolutional layer
            target_layer = self.model.base_model.features.denseblock4
            
            # Get weights of the classifier
            fc_weights = self.model.classifier[-1].weight.data
            
        else:
            raise ValueError(f"Unsupported model architecture: {self.model.model_name}")
        
        return target_layer, fc_weights
    
    def _get_features(self, image):
        """
        Get the feature maps from the target layer.
        
        Args:
            image (torch.Tensor): Input image tensor of shape (1, C, H, W)
            
        Returns:
            torch.Tensor: Feature maps from the target layer
        """
        # Create a hook to get the output of the target layer
        features = []
        
        def hook(module, input, output):
            features.append(output.detach())
        
        # Register the hook
        handle = self.target_layer.register_forward_hook(hook)
        
        # Forward pass
        with torch.no_grad():
            self.model(image)
        
        # Remove the hook
        handle.remove()
        
        return features[0]
    
    def generate_cam(self, image, target_class=None):
        """
        Generate a CAM for the target class.
        
        Args:
            image (torch.Tensor): Input image tensor of shape (1, C, H, W)
            target_class (int, optional): Target class index
                If None, uses the predicted class
                
        Returns:
            numpy.ndarray: CAM heatmap of shape (H', W'), range [0, 1]
        """
        # Move image to device
        image = image.to(self.device)
        
        # If target_class is None, use the predicted class
        if target_class is None:
            with torch.no_grad():
                output = self.model(image)
                target_class = output.argmax(dim=1).item()
        
        # Get feature maps from the target layer
        features = self._get_features(image)
        
        # Get weights for the target class
        class_weights = self.fc_weights[target_class].cpu().numpy()
        
        # Compute weighted sum of feature maps
        cam = np.zeros(features.shape[2:], dtype=np.float32)
        
        for i, w in enumerate(class_weights):
            # For each channel, multiply by the corresponding weight
            cam += w * features[0, i].cpu().numpy()
        
        # Apply ReLU to focus on positive contributions
        cam = np.maximum(cam, 0)
        
        # Normalize
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-8)
        
        return cam
    
    def visualize(self, image, original_image, target_class=None, colormap=COLORMAP,
                 alpha=OVERLAY_ALPHA, figsize=(12, 4), save_path=None):
        """
        Visualize CAM as a heatmap overlay on the original image.
        
        Args:
            image (torch.Tensor): Input image tensor of shape (1, C, H, W)
            original_image (numpy.ndarray): Original image array of shape (H, W, 3)
            target_class (int, optional): Target class index
                If None, uses the predicted class
            colormap (str): Matplotlib colormap to use for the heatmap
            alpha (float): Transparency of the heatmap overlay
            figsize (tuple): Figure size
            save_path (str, optional): Path to save the visualization
            
        Returns:
            tuple: (heatmap, overlaid_image)
        """
        # Generate CAM
        cam = self.generate_cam(image, target_class)
        
        # Get the predicted class if target_class is None
        if target_class is None:
            with torch.no_grad():
                output = self.model(image.to(self.device))
                target_class = output.argmax(dim=1).item()
                confidence = F.softmax(output, dim=1)[0, target_class].item()
        else:
            with torch.no_grad():
                output = self.model(image.to(self.device))
                confidence = F.softmax(output, dim=1)[0, target_class].item()
        
        # Convert heatmap to RGB using the specified colormap
        cmap = plt.get_cmap(colormap)
        heatmap_rgb = cmap(cam)[:, :, :3]  # Remove alpha channel
        
        # Resize heatmap to match original image
        heatmap_resized = cv2.resize(
            cam, (original_image.shape[1], original_image.shape[0])
        )
        
        # Convert resized heatmap to RGB
        heatmap_rgb_resized = cmap(heatmap_resized)[:, :, :3]
        
        # Create a blended overlay image
        overlaid_image = (1 - alpha) * original_image + alpha * heatmap_rgb_resized
        
        # Clip values to valid range
        overlaid_image = np.clip(overlaid_image, 0, 1)
        
        # Create a figure with subplots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
        
        # Plot original image
        ax1.imshow(original_image)
        ax1.set_title(f"Original Image\nClass: {target_class} ({confidence:.2f})")
        ax1.axis('off')
        
        # Plot heatmap
        ax2.imshow(heatmap_rgb, cmap=colormap)
        ax2.set_title("CAM Heatmap")
        ax2.axis('off')
        
        # Plot overlay
        ax3.imshow(overlaid_image)
        ax3.set_title("Overlay")
        ax3.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
        
        plt.show()
        
        return cam, overlaid_image


def apply_cam(model, image_tensor, original_image, target_class=None, save_path=None, device=None):
    """
    Apply CAM to an image.
    
    Args:
        model (torch.nn.Module): The model to visualize
        image_tensor (torch.Tensor): Input image tensor of shape (1, C, H, W)
        original_image (numpy.ndarray): Original image array of shape (H, W, 3)
        target_class (int, optional): Target class index
            If None, uses the predicted class
        save_path (str, optional): Path to save the visualization
        device (torch.device, optional): Device to run the model on
        
    Returns:
        tuple: (heatmap, overlaid_image)
    """
    # Create CAM object
    cam = CAM(model, device)
    
    # Visualize
    heatmap, overlaid_image = cam.visualize(
        image_tensor, original_image, target_class, save_path=save_path
    )
    
    return heatmap, overlaid_image