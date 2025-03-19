"""
Fixed implementation of Grad-CAM for skin cancer classification model.

This module implements a more robust version of Gradient-weighted Class 
Activation Mapping (Grad-CAM) for visualizing important regions in an image.
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
    
    def __init__(self, model, target_layer=None, device=None):
        """
        Initialize GradCAM.
        
        Args:
            model (torch.nn.Module): The model to visualize
            target_layer (torch.nn.Module, optional): The convolutional layer to use
                If None, tries to use the layer specified in GRADCAM_LAYER
            device (torch.device, optional): Device to run the model on
        """
        self.model = model
        self.model.eval()
        
        # Set device
        if device is None:
            self.device = next(model.parameters()).device
        else:
            self.device = device
            self.model = self.model.to(self.device)
        
        # Set target layer
        if target_layer is None:
            if GRADCAM_LAYER == "auto":
                # Try to automatically find the target layer
                target_layer = self._find_target_layer()
            else:
                # Try to get the specified layer
                try:
                    if hasattr(model, 'model_name'):
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
        print(f"Using target layer: {self.target_layer}")
        
        # Initialize variables to store activations and gradients
        self.activations = None
        self.gradients = None
        
        # Register hooks
        self.forward_handle = self.target_layer.register_forward_hook(self._forward_hook)
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
                if hasattr(self.model.base_model.features, '_modules'):
                    # Get the last convolutional layer
                    modules = list(self.model.base_model.features._modules.values())
                    for module in reversed(modules):
                        if isinstance(module, torch.nn.Conv2d):
                            target_layer = module
                            break
                else:
                    target_layer = self.model.base_model.features[-1]
            # DenseNet
            elif hasattr(self.model.base_model, 'features') and hasattr(self.model.base_model.features, 'denseblock4'):
                target_layer = self.model.base_model.features.denseblock4

        # Generic fallback: find the last convolutional layer in the model
        if target_layer is None:
            for name, module in reversed(list(self.model.named_modules())):
                if isinstance(module, torch.nn.Conv2d):
                    target_layer = module
                    print(f"Automatically selected layer: {name}")
                    break
        
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
        # Ensure input is on the same device as model
        input_tensor = input_tensor.to(self.device)
        
        # Reset gradients
        self.model.zero_grad()
        
        # Reset recorded values
        self.activations = None
        self.gradients = None
        
        # Forward pass
        output = self.model(input_tensor)
        
        # If target_class is None, use the predicted class
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Get the score for the target class
        target_score = output[0, target_class]
        
        # Backward pass to get gradients
        target_score.backward()
        
        # Check if hooks captured values
        if self.gradients is None:
            print("Warning: No gradients were captured. Check model architecture and hooks.")
            return np.zeros((input_tensor.shape[2], input_tensor.shape[3]))
        
        if self.activations is None:
            print("Warning: No activations were captured. Check model architecture and hooks.")
            return np.zeros((input_tensor.shape[2], input_tensor.shape[3]))
        
        # Get the average gradient for each feature map
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        
        # Create a weighted combination of the activation maps
        # Start with a copy of the activations
        weighted_activations = self.activations.clone()
        
        # Weight each channel of the activations by the corresponding gradient
        for i in range(pooled_gradients.size(0)):
            weighted_activations[0, i, :, :] *= pooled_gradients[i]
        
        # Average the weighted feature maps
        heatmap = torch.mean(weighted_activations, dim=1).squeeze().cpu().numpy()
        
        # ReLU to only keep positive contributions
        heatmap = np.maximum(heatmap, 0)
        
        # Normalize heatmap
        if np.max(heatmap) > 0:  # Avoid division by zero
            heatmap = heatmap / np.max(heatmap)
        
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
                output = self.model(input_tensor.to(self.device))
                target_class = output.argmax(dim=1).item()
                confidence = F.softmax(output, dim=1)[0, target_class].item()
        else:
            with torch.no_grad():
                output = self.model(input_tensor.to(self.device))
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


def simple_occlusion_map(model, input_tensor, original_image, target_class=None, 
                        stride=8, patch_size=24, save_path=None, device=None):
    """
    A simple occlusion-based approach to generate heatmaps.
    This method works by occluding parts of the image and seeing how the prediction changes.
    
    Args:
        model (torch.nn.Module): The model to visualize
        input_tensor (torch.Tensor): Input image tensor of shape (1, C, H, W)
        original_image (numpy.ndarray): Original image array of shape (H, W, 3)
        target_class (int, optional): Target class index
            If None, uses the predicted class
        stride (int): Step size for occlusion window
        patch_size (int): Size of the occlusion patch
        save_path (str, optional): Path to save the visualization
        device (torch.device, optional): Device to use for computation
        
    Returns:
        tuple: (heatmap, overlaid_image)
    """
    if device is None:
        device = next(model.parameters()).device
    
    input_tensor = input_tensor.to(device)
    model.eval()
    
    # Get original prediction and class
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        if target_class is None:
            target_class = outputs.argmax(dim=1).item()
        original_score = probs[0, target_class].item()
    
    # Get image dimensions
    _, c, h, w = input_tensor.shape
    
    # Initialize heatmap
    heatmap = np.zeros((h, w))
    count = np.zeros((h, w))
    
    # Create an occlusion patch (gray)
    occlusion = torch.ones((1, c, patch_size, patch_size)).to(device) * 0.5
    
    # Iterate over the image with the occlusion window
    for i in range(0, h - patch_size + 1, stride):
        for j in range(0, w - patch_size + 1, stride):
            # Make a copy of the input tensor
            occluded = input_tensor.clone()
            
            # Apply occlusion
            occluded[0, :, i:i+patch_size, j:j+patch_size] = occlusion
            
            # Get prediction for occluded image
            with torch.no_grad():
                outputs = model(occluded)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                occluded_score = probs[0, target_class].item()
            
            # Calculate importance: how much score changes when occluded
            diff = original_score - occluded_score
            
            # Update heatmap (higher diff = more important region)
            heatmap[i:i+patch_size, j:j+patch_size] += diff
            count[i:i+patch_size, j:j+patch_size] += 1
    
    # Average overlapping regions
    heatmap = np.divide(heatmap, count, out=np.zeros_like(heatmap), where=count!=0)
    
    # Normalize heatmap to [0, 1]
    heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-8)
    
    # Resize heatmap to match original image
    heatmap_resized = cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0]))
    
    # Create RGB heatmap
    cmap = plt.get_cmap(COLORMAP)
    heatmap_rgb = cmap(heatmap)[:, :, :3]
    heatmap_rgb_resized = cmap(heatmap_resized)[:, :, :3]
    
    # Create overlay image
    overlaid_image = (1 - OVERLAY_ALPHA) * original_image + OVERLAY_ALPHA * heatmap_rgb_resized
    overlaid_image = np.clip(overlaid_image, 0, 1)
    
    # Create visualization
    plt.figure(figsize=(12, 4))
    
    # Original image
    plt.subplot(1, 3, 1)
    plt.imshow(original_image)
    plt.title(f"Original Image\nClass: {target_class} ({original_score:.2f})")
    plt.axis('off')
    
    # Heatmap
    plt.subplot(1, 3, 2)
    plt.imshow(heatmap_rgb, cmap=COLORMAP)
    plt.title("Occlusion Heatmap")
    plt.axis('off')
    
    # Overlay
    plt.subplot(1, 3, 3)
    plt.imshow(overlaid_image)
    plt.title("Overlay")
    plt.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    
    plt.show()
    
    return heatmap, overlaid_image


def apply_gradcam(model, image_tensor, original_image, target_class=None, 
                 target_layer=None, save_path=None, device=None):
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
        device (torch.device, optional): Device to run the model on
        
    Returns:
        tuple: (heatmap, overlaid_image)
    """
    try:
        # Try to create GradCAM object
        print("Initializing Grad-CAM...")
        grad_cam = GradCAM(model, target_layer, device)
        
        # Visualize
        print("Generating Grad-CAM visualization...")
        heatmap, overlaid_image = grad_cam.visualize(
            image_tensor, original_image, target_class, save_path=save_path
        )
        
        return heatmap, overlaid_image
    except Exception as e:
        print(f"Error in Grad-CAM: {e}")
        print("Falling back to occlusion map...")
        return simple_occlusion_map(
            model, image_tensor, original_image, 
            target_class=target_class, save_path=save_path, device=device
        )