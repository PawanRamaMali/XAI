"""
SHAP (SHapley Additive exPlanations) implementation for skin cancer classification model.

This module implements SHAP for attributing the prediction of a deep network
to its input features, using Shapley values from game theory.

Reference:
    Lundberg and Lee, "A Unified Approach to Interpreting Model Predictions",
    https://arxiv.org/abs/1705.07874
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import shap

import torch
import torch.nn.functional as F

import sys
sys.path.append('..')
from config import SHAP_NUM_SAMPLES, DPI


class ShapExplainer:
    """
    SHAP explainer for image classification models.
    
    This class implements SHAP (SHapley Additive exPlanations) to explain
    predictions by computing the contribution of each feature to the prediction.
    """
    
    def __init__(self, model, device=None):
        """
        Initialize ShapExplainer.
        
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
    
    def predict_fn(self, images):
        """
        Prediction function to be used by SHAP.
        
        Args:
            images (numpy.ndarray): Batch of images of shape (N, C, H, W)
            
        Returns:
            numpy.ndarray: Predicted probabilities of shape (N, num_classes)
        """
        # Convert to PyTorch tensor if it's not already
        if not isinstance(images, torch.Tensor):
            images = torch.tensor(images, device=self.device, dtype=torch.float32)
            
        # Move to device
        images = images.to(self.device)
        
        # Get predictions
        with torch.no_grad():
            logits = self.model(images)
            probs = F.softmax(logits, dim=1)
            
        return probs.cpu().numpy()
    
    def explain(self, image, background=None, target_class=None, num_samples=SHAP_NUM_SAMPLES):
        """
        Generate a SHAP explanation for the given image.
        
        Args:
            image (torch.Tensor): Input image tensor of shape (1, C, H, W)
            background (torch.Tensor, optional): Background images tensor of shape (N, C, H, W)
                If None, uses a black image (all zeros)
            target_class (int, optional): Target class index
                If None, uses the predicted class
            num_samples (int): Number of samples to use for SHAP
            
        Returns:
            numpy.ndarray: SHAP values of shape (1, C, H, W)
        """
        # Move image to device
        image = image.to(self.device)
        
        # If target_class is None, use the predicted class
        if target_class is None:
            with torch.no_grad():
                output = self.model(image)
                target_class = output.argmax(dim=1).item()
        
        # If background is None, use a black image
        if background is None:
            # Create a minimal background set (just a black image)
            background = torch.zeros((1,) + image.shape[1:], device=self.device)
        else:
            background = background.to(self.device)
        
        # Move everything to CPU for SHAP
        image_cpu = image.cpu()
        background_cpu = background.cpu()
        
        # Create a SHAP DeepExplainer
        explainer = shap.DeepExplainer(
            model=lambda x: self.predict_fn(x),
            data=background_cpu
        )
        
        # Compute SHAP values
        shap_values = explainer.shap_values(
            image_cpu,
            nsamples=num_samples
        )
        
        # Extract SHAP values for the target class
        shap_values_target = shap_values[target_class]
        
        return shap_values_target
    
    def visualize(self, image, original_image, background=None, target_class=None, 
                 num_samples=SHAP_NUM_SAMPLES, figsize=(15, 5), save_path=None):
        """
        Visualize a SHAP explanation for the given image.
        
        Args:
            image (torch.Tensor): Input image tensor of shape (1, C, H, W)
            original_image (numpy.ndarray): Original image array of shape (H, W, 3)
            background (torch.Tensor, optional): Background images tensor
            target_class (int, optional): Target class index
                If None, uses the predicted class
            num_samples (int): Number of samples to use for SHAP
            figsize (tuple): Figure size
            save_path (str, optional): Path to save the visualization
            
        Returns:
            numpy.ndarray: SHAP values
        """
        # Get the predicted class and confidence if target_class is None
        if target_class is None:
            with torch.no_grad():
                output = self.model(image.to(self.device))
                target_class = output.argmax(dim=1).item()
                confidence = F.softmax(output, dim=1)[0, target_class].item()
        else:
            with torch.no_grad():
                output = self.model(image.to(self.device))
                confidence = F.softmax(output, dim=1)[0, target_class].item()
        
        # Generate explanation
        shap_values = self.explain(image, background, target_class, num_samples)
        
        # Create a custom colormap for SHAP visualizations
        colors = []
        for l, c in zip(np.linspace(0, 1, 100), plt.cm.coolwarm(np.linspace(0, 1, 100))):
            colors.append((l, c[0], c[1], c[2]))
        custom_cmap = LinearSegmentedColormap.from_list("custom", colors)
        
        # Compute absolute values and sum across channels for a heatmap
        shap_abs = np.abs(shap_values).sum(axis=1)[0]
        
        # Normalize for visualization
        shap_norm = shap_abs / (shap_abs.max() + 1e-10)
        
        # Resize to match original image
        from skimage.transform import resize
        shap_resized = resize(shap_norm, (original_image.shape[0], original_image.shape[1]))
        
        # Create visualization
        plt.figure(figsize=figsize)
        
        plt.subplot(1, 3, 1)
        plt.imshow(original_image)
        plt.title(f"Original Image\nClass: {target_class} ({confidence:.2f})")
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(shap_resized, cmap=custom_cmap)
        plt.title("SHAP Attribution Magnitude")
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        # Create a blended image
        cmap_img = plt.cm.coolwarm(shap_resized)[:,:,:3]
        blended = 0.7 * original_image + 0.3 * cmap_img
        blended = np.clip(blended, 0, 1)
        
        plt.imshow(blended)
        plt.title("SHAP Overlay")
        plt.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
        
        plt.show()
        
        # Also use SHAP's built-in visualization
        plt.figure(figsize=figsize)
        
        # Converting for SHAP's visualization
        img_for_shap = np.transpose(image.cpu().numpy()[0], (1, 2, 0))
        
        shap.image_plot(
            shap_values=shap_values,
            pixel_values=img_for_shap,
            labels=[f"Class {target_class}"]
        )
        
        if save_path:
            new_path = save_path.replace('.png', '_shap_native.png')
            plt.savefig(new_path, dpi=DPI, bbox_inches='tight')
        
        return shap_values


def apply_shap(model, image_tensor, original_image, target_class=None, 
              num_samples=SHAP_NUM_SAMPLES, save_path=None, device=None):
    """
    Apply SHAP to an image.
    
    Args:
        model (torch.nn.Module): The model to explain
        image_tensor (torch.Tensor): Input image tensor of shape (1, C, H, W)
        original_image (numpy.ndarray): Original image array of shape (H, W, 3)
        target_class (int, optional): Target class index
            If None, uses the predicted class
        num_samples (int): Number of samples to use for SHAP
        save_path (str, optional): Path to save the visualization
        device (torch.device, optional): Device to run the model on
        
    Returns:
        numpy.ndarray: SHAP values
    """
    # Create SHAP explainer
    explainer = ShapExplainer(model, device)
    
    # Visualize
    shap_values = explainer.visualize(
        image_tensor,
        original_image,
        target_class=target_class,
        num_samples=num_samples,
        save_path=save_path
    )
    
    return shap_values