"""
Grad-CAM implementation for skin cancer classification model.

This module implements Gradient-weighted Class Activation Mapping (Grad-CAM)
for visualizing the regions of an input image that are important for predictions.

Reference:
    Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via 
    Gradient-based Localization", https://arxiv.org/abs/1610.02391
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2

import torch
import torch.nn.functional as F

import sys
sys.path.append('..')
from config import GRADCAM_LAYER, COLORMAP, OVERLAY_ALPHA, DPI


class GradCAM:
    """
    Grad-CAM visualization for CNN models.
    
    This class implements Grad-CAM, which uses the gradients of a target class
    flowing into the final convolutional layer to produce a coarse localization map
    highlighting the important regions in the image for predicting the target class.
    """
    
    def __init__(self, model, target_layer=None):
        """
        Initialize GradCAM.
        
        Args:
            model (torch.nn.Module): The model to visualize
            target_layer (torch.nn.Module, optional): The convolutional layer to use
                If None, tries to use the layer specified in GRADCAM_LAYER
        """
        self.model = model
        self.model.eval()
        
        # Set target layer
        if target_layer is None:
            if GRADCAM_LAYER == "auto":
                # Try to automatically find the target layer
                target_layer = self._find_target_layer()
            else:
                # Try to get the specified layer
                try:
                    if 'resnet' in model.model_name:
                        if GRADCAM_LAYER == "layer4":
                            target_layer = model.base_model.layer4[-1]
                        elif GRADCAM_LAYER == "layer3":
                            target_layer = model.base_model.layer3[-1]
                    elif 'efficientnet' in model.model_name:
                        if GRADCAM_LAYER == "features":
                            target_layer = model.base_model.features[-1]
                    elif 'densenet' in model.model_name:
                        if GRADCAM_LAYER == "features":
                            target_layer = model.base_model.features.denseblock4
                except Exception as e:
                    print(f"Failed to get target layer: {e}")
                    target_layer = self._find_target_layer()
                    
        if target_layer is None:
            raise ValueError(
                "Could not determine target layer for Grad-CAM. "
                "Please specify a target layer manually."
            )
            
        self.target_layer = target_layer
        
        # Register hooks to record gradients and activations
        self.gradients = []
        self.activations = []
        
        # Register forward hook
        self.forward_handle = self.target_layer.register_forward_hook(self._forward_hook)
        
        # Register backward hook
        self.backward_handle = self.target_layer.register_full_backward_hook(self._backward_hook)
    
    def _find_target_layer(self):
        """
        Attempt to automatically find a suitable convolutional layer for Grad-CAM.
        
        Returns:
            torch.nn.Module: The target layer, or None if not found
        """
        # Try to find the last convolutional layer
        if hasattr(self.model, 'get_target_layer'):
            return self.model.get_target_layer()
            
        # If the model doesn't have a get_target_layer method, try to find it
        target_layer = None
        
        # For ResNet-like models
        if hasattr(self.model, 'base_model'):
            # ResNet
            if hasattr(self.model.base_model, 'layer4'):
                target_layer = self.model.base_model.layer4[-1]
            # EfficientNet
            elif hasattr(self.model.base_model, 'features'):
                target_layer = self.model.base_model.features[-1]
            # DenseNet
            elif hasattr(self.model.base_model, 'features') and hasattr(self.model.base_model.features, 'denseblock4'):
                target_layer = self.model.base_model.features.denseblock4
        
        return target_layer
    
    def _forward_hook(self, module, input, output):
        """Hook for recording the activations of the target layer."""
        self.activations = output.detach()
    
    def _backward_hook(self, module, grad_input, grad_output):
        """Hook for recording the gradients at the target layer."""
        self.gradients = grad_output[0].detach()
    
    def __del__(self):
        """Clean up by removing hooks when the object is deleted."""
        try:
            self.forward_handle.remove()
            self.backward_handle.remove()
        except:
            pass
    
    def generate_cam(self, input_tensor, target_class=None):
        """
        Generate a Grad-CAM visualization for the target class.
        
        Args:
            input_tensor (torch.Tensor): Input image tensor of shape (1, C, H, W)
            target_class (int, optional): Target class index
                If None, uses the predicted class
                
        Returns:
            numpy.ndarray: Grad-CAM heatmap of shape (H, W), range [0, 1]
        """
        # Reset gradients
        self.model.zero_grad()
        
        # Forward pass
        output = self.model(input_tensor)
        
        # If target_class is None, use the predicted class
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Get the score for the target class
        target_score = output[0, target_class]
        
        # Backward pass to get gradients
        target_score.backward()
        
        # Get the average gradient for each feature map
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        
        # Weight the feature maps by the gradients
        for i in range(self.activations.size(1)):
            self.activations[:, i, :, :] *= pooled_gradients[i]
        
        # Average the weighted feature maps
        heatmap = torch.mean(self.activations, dim=1).squeeze().cpu().detach().numpy()
        
        # ReLU to only keep positive contributions
        heatmap = np.maximum(heatmap, 0)
        
        # Normalize heatmap
        heatmap = heatmap / (np.max(heatmap) + 1e-10)
        
        return heatmap
    
    def visualize(self, input_tensor, original_image, target_class=None, colormap=COLORMAP,
                 alpha=OVERLAY_ALPHA, figsize=(12, 4), save_path=None):
        """
        Visualize Grad-CAM as a heatmap overlay on the original image.
        
        Args:
            input_tensor (torch.Tensor): Input image tensor of shape (1, C, H, W)
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
        # Generate Grad-CAM
        heatmap = self.generate_cam(input_tensor, target_class)
        
        # Get the predicted class if target_class is None
        if target_class is None:
            with torch.no_grad():
                output = self.model(input_tensor)
                target_class = output.argmax(dim=1).item()
                confidence = F.softmax(output, dim=1)[0, target_class].item()
        else:
            with torch.no_grad():
                output = self.model(input_tensor)
                confidence = F.softmax(output, dim=1)[0, target_class].item()
        
        # Convert heatmap to RGB using the specified colormap
        cmap = plt.get_cmap(colormap)
        heatmap_rgb = cmap(heatmap)[:, :, :3]  # Remove alpha channel
        
        # Resize heatmap to match original image
        heatmap_resized = cv2.resize(
            heatmap, (original_image.shape[1], original_image.shape[0])
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
        ax2.set_title("Grad-CAM Heatmap")
        ax2.axis('off')
        
        # Plot overlay
        ax3.imshow(overlaid_image)
        ax3.set_title("Overlay")
        ax3.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
        
        plt.show()
        
        return heatmap, overlaid_image


def apply_gradcam(model, image_tensor, original_image, target_class=None, 
                 target_layer=None, save_path=None):
    """
    Apply Grad-CAM to an image.
    
    Args:
        model (torch.nn.Module): The model to visualize
        image_tensor (torch.Tensor): Input image tensor of shape (1, C, H, W)
        original_image (numpy.ndarray): Original image array of shape (H, W, 3)
        target_class (int, optional): Target class index
            If None, uses the predicted class
        target_layer (torch.nn.Module, optional): The convolutional layer to use
        save_path (str, optional): Path to save the visualization
        
    Returns:
        tuple: (heatmap, overlaid_image)
    """
    # Create GradCAM object
    grad_cam = GradCAM(model, target_layer)
    
    # Visualize
    heatmap, overlaid_image = grad_cam.visualize(
        image_tensor, original_image, target_class, save_path=save_path
    )
    
    return heatmap, overlaid_image