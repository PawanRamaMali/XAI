"""
Captum-based implementation of Grad-CAM for skin cancer classification model.

This module implements Gradient-weighted Class Activation Mapping (Grad-CAM) using
the Captum library, which provides more robust implementations of explainability
techniques for PyTorch models.
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2

import torch
import torch.nn.functional as F

# Import Captum components
from captum.attr import LayerGradCam, LayerAttribution

import sys
sys.path.append('..')
from config import COLORMAP, OVERLAY_ALPHA, DPI


def find_target_layer(model):
    """
    Attempt to automatically find a suitable convolutional layer for Grad-CAM.
    
    Args:
        model (torch.nn.Module): The model to examine
        
    Returns:
        tuple: (target_layer, layer_name)
    """
    # Try to find the last convolutional layer
    target_layer = None
    layer_name = None
    
    # If the model has a get_target_layer method, use it
    if hasattr(model, 'get_target_layer'):
        return model.get_target_layer(), "model_defined_target_layer"
    
    # For ResNet-like models
    if hasattr(model, 'base_model'):
        # ResNet
        if hasattr(model.base_model, 'layer4'):
            target_layer = model.base_model.layer4[-1]
            layer_name = "base_model.layer4[-1]"
        # EfficientNet
        elif hasattr(model.base_model, 'features'):
            if hasattr(model.base_model.features, '_modules'):
                # Get the last convolutional layer
                modules = list(model.base_model.features._modules.values())
                for i, module in enumerate(reversed(modules)):
                    if isinstance(module, torch.nn.Conv2d):
                        target_layer = module
                        layer_name = f"base_model.features[{len(modules)-i-1}]"
                        break
            else:
                target_layer = model.base_model.features[-1]
                layer_name = "base_model.features[-1]"
        # DenseNet
        elif hasattr(model.base_model, 'features') and hasattr(model.base_model.features, 'denseblock4'):
            target_layer = model.base_model.features.denseblock4
            layer_name = "base_model.features.denseblock4"

    # Generic fallback: find the last convolutional layer in the model
    if target_layer is None:
        for name, module in reversed(list(model.named_modules())):
            if isinstance(module, torch.nn.Conv2d):
                target_layer = module
                layer_name = name
                print(f"Automatically selected layer: {name}")
                break
    
    return target_layer, layer_name


def apply_captum_gradcam(model, image_tensor, original_image, target_class=None, 
                         target_layer=None, save_path=None, device=None):
    """
    Apply Grad-CAM to an image using Captum library.
    
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
    # Set device
    if device is None:
        device = next(model.parameters()).device
    
    model = model.to(device)
    image_tensor = image_tensor.to(device)
    
    # Set model to eval mode
    model.eval()
    
    # Get prediction if target_class is None
    if target_class is None:
        with torch.no_grad():
            output = model(image_tensor)
            target_class = output.argmax(dim=1).item()
            confidence = F.softmax(output, dim=1)[0, target_class].item()
    else:
        with torch.no_grad():
            output = model(image_tensor)
            confidence = F.softmax(output, dim=1)[0, target_class].item()
    
    # Find target layer if not provided
    if target_layer is None:
        print("Finding suitable target layer for Grad-CAM...")
        target_layer, layer_name = find_target_layer(model)
        print(f"Using target layer: {layer_name}")
    
    try:
        # Create Captum Grad-CAM instance
        grad_cam = LayerGradCam(model, target_layer)
        
        # Compute attributions
        print("Computing Grad-CAM attributions...")
        attributions = grad_cam.attribute(image_tensor, target=target_class)
        
        # Upsample attributions to input size
        heatmap = LayerAttribution.interpolate(attributions, image_tensor.shape[2:])
        
        # Convert to numpy for visualization
        heatmap = heatmap.squeeze().cpu().detach().numpy()
        
        # Apply ReLU to highlight only positive contributions
        heatmap = np.maximum(heatmap, 0)
        
        # Normalize heatmap
        if np.max(heatmap) > 0:  # Avoid division by zero
            heatmap = heatmap / np.max(heatmap)
        
        # Create RGB heatmap using specified colormap
        cmap = plt.get_cmap(COLORMAP)
        heatmap_rgb = cmap(heatmap)[:, :, :3]  # Remove alpha channel
        
        # Resize heatmap to match original image
        heatmap_resized = cv2.resize(
            heatmap, (original_image.shape[1], original_image.shape[0])
        )
        
        # Convert resized heatmap to RGB
        heatmap_rgb_resized = cmap(heatmap_resized)[:, :, :3]
        
        # Create a blended overlay image
        overlaid_image = (1 - OVERLAY_ALPHA) * original_image + OVERLAY_ALPHA * heatmap_rgb_resized
        
        # Clip values to valid range
        overlaid_image = np.clip(overlaid_image, 0, 1)
        
        # Create visualization
        plt.figure(figsize=(12, 4))
        
        # Original image
        plt.subplot(1, 3, 1)
        plt.imshow(original_image)
        plt.title(f"Original Image\nClass: {target_class} ({confidence:.2f})")
        plt.axis('off')
        
        # Heatmap
        plt.subplot(1, 3, 2)
        plt.imshow(heatmap, cmap=COLORMAP)
        plt.title("Captum Grad-CAM Heatmap")
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
        
    except Exception as e:
        print(f"Error in Captum Grad-CAM: {e}")
        from explanation.gradcam import simple_occlusion_map
        
        print("Falling back to occlusion map...")
        return simple_occlusion_map(
            model, image_tensor, original_image, 
            target_class=target_class, save_path=save_path, device=device
        )


# Example usage:
# heatmap, overlay = apply_captum_gradcam(model, image_tensor, original_image, target_class=1)