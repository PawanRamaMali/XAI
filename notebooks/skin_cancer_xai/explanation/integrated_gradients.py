"""
Integrated Gradients implementation for skin cancer classification model.

This module implements the Integrated Gradients method for attributing
the prediction of a deep network to its input features.

Reference:
    Sundararajan et al., "Axiomatic Attribution for Deep Networks",
    https://arxiv.org/abs/1703.01365
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2

import torch
import torch.nn.functional as F

import sys
sys.path.append('..')
from config import IG_STEPS, COLORMAP, DPI


class IntegratedGradients:
    """
    Integrated Gradients explainer for image classification models.
    
    This class implements Integrated Gradients, which attributes
    the prediction of a deep network to its input features by
    integrating the gradients with respect to the input along
    a straight line from a baseline to the input.
    """
    
    def __init__(self, model, device=None):
        """
        Initialize IntegratedGradients.
        
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
    
    def _get_gradients(self, image, target_class):
        """
        Compute gradients of the model output with respect to the input image.
        
        Args:
            image (torch.Tensor): Input image tensor of shape (1, C, H, W)
            target_class (int): Target class index
            
        Returns:
            torch.Tensor: Gradients of shape (1, C, H, W)
        """
        # Enable gradient computation
        image.requires_grad = True
        
        # Forward pass
        output = self.model(image)
        
        # Get target class score
        score = output[0, target_class]
        
        # Backward pass to get gradients
        self.model.zero_grad()
        score.backward()
        
        # Get gradients with respect to input
        grads = image.grad.clone()
        
        # Cleanup
        image.requires_grad = False
        image.grad = None
        
        return grads
    
    def explain(self, image, baseline=None, target_class=None, steps=IG_STEPS):
        """
        Generate an Integrated Gradients explanation for the given image.
        
        Args:
            image (torch.Tensor): Input image tensor of shape (1, C, H, W)
            baseline (torch.Tensor, optional): Baseline image tensor
                If None, uses a black image (all zeros)
            target_class (int, optional): Target class index
                If None, uses the predicted class
            steps (int): Number of steps in the gradient path
            
        Returns:
            torch.Tensor: Attribution map of shape (1, C, H, W)
        """
        # Move image to device
        image = image.to(self.device)
        
        # If target_class is None, use the predicted class
        if target_class is None:
            with torch.no_grad():
                output = self.model(image)
                target_class = output.argmax(dim=1).item()
        
        # If baseline is None, use a black image
        if baseline is None:
            baseline = torch.zeros_like(image, device=self.device)
        else:
            baseline = baseline.to(self.device)
        
        # Generate alphas for interpolation
        alphas = torch.linspace(0, 1, steps, device=self.device)
        
        # Initialize accumulator for gradients
        integrated_gradients = torch.zeros_like(image, device=self.device)
        
        # Compute path integral
        for alpha in alphas:
            # Interpolate between baseline and image
            interpolated = baseline + alpha * (image - baseline)
            interpolated.requires_grad = True
            
            # Compute gradients
            grads = self._get_gradients(interpolated, target_class)
            
            # Accumulate gradients
            integrated_gradients += grads
            
        # Scale integrated gradients by path length
        integrated_gradients *= (image - baseline) / steps
        
        return integrated_gradients
    
    def visualize(self, image, original_image, baseline=None, target_class=None, 
                 steps=IG_STEPS, figsize=(15, 5), save_path=None):
        """
        Visualize an Integrated Gradients explanation for the given image.
        
        Args:
            image (torch.Tensor): Input image tensor of shape (1, C, H, W)
            original_image (numpy.ndarray): Original image array of shape (H, W, 3)
            baseline (torch.Tensor, optional): Baseline image tensor
                If None, uses a black image (all zeros)
            target_class (int, optional): Target class index
                If None, uses the predicted class
            steps (int): Number of steps in the gradient path
            figsize (tuple): Figure size
            save_path (str, optional): Path to save the visualization
            
        Returns:
            numpy.ndarray: Attribution heatmap
        """
        # Generate explanation
        attributions = self.explain(image, baseline, target_class, steps)
        
        # Get the predicted class and confidence
        if target_class is None:
            with torch.no_grad():
                output = self.model(image.to(self.device))
                target_class = output.argmax(dim=1).item()
                confidence = F.softmax(output, dim=1)[0, target_class].item()
        else:
            with torch.no_grad():
                output = self.model(image.to(self.device))
                confidence = F.softmax(output, dim=1)[0, target_class].item()
        
        # Convert attributions to numpy
        attributions = attributions.cpu().numpy()[0]
        
        # Sum attributions across color channels
        attribution_map = np.sum(np.abs(attributions), axis=0)
        
        # Normalize attribution map
        attribution_map = (attribution_map - attribution_map.min()) / (attribution_map.max() - attribution_map.min() + 1e-8)
        
        # Create heatmap
        heatmap = cv2.applyColorMap(
            np.uint8(255 * attribution_map), 
            getattr(cv2, f'COLORMAP_{COLORMAP.upper()}')
        )
        
        # Convert to RGB
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Resize heatmap to match original image
        heatmap = cv2.resize(
            heatmap, (original_image.shape[1], original_image.shape[0])
        )
        
        # Create blended image
        blended = 0.7 * original_image + 0.3 * heatmap / 255.0
        blended = np.clip(blended, 0, 1)
        
        # Create visualization
        plt.figure(figsize=figsize)
        
        plt.subplot(1, 3, 1)
        plt.imshow(original_image)
        plt.title(f"Original Image\nClass: {target_class} ({confidence:.2f})")
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(heatmap)
        plt.title("Attribution Heatmap")
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.imshow(blended)
        plt.title("Blended Image")
        plt.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
        
        plt.show()
        
        return attribution_map


def apply_integrated_gradients(model, image_tensor, original_image, target_class=None, 
                              steps=IG_STEPS, save_path=None, device=None):
    """
    Apply Integrated Gradients to an image.
    
    Args:
        model (torch.nn.Module): The model to explain
        image_tensor (torch.Tensor): Input image tensor of shape (1, C, H, W)
        original_image (numpy.ndarray): Original image array of shape (H, W, 3)
        target_class (int, optional): Target class index
            If None, uses the predicted class
        steps (int): Number of steps in the gradient path
        save_path (str, optional): Path to save the visualization
        device (torch.device, optional): Device to run the model on
        
    Returns:
        numpy.ndarray: Attribution heatmap
    """
    # Create Integrated Gradients explainer
    explainer = IntegratedGradients(model, device)
    
    # Visualize
    attribution_map = explainer.visualize(
        image_tensor,
        original_image,
        target_class=target_class,
        steps=steps,
        save_path=save_path
    )
    
    return attribution_map