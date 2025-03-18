"""
LIME explainer for skin cancer classification model.

This module implements the Local Interpretable Model-agnostic Explanations (LIME)
method for visualizing which parts of an image are most influential for 
a given prediction.

Reference:
    Ribeiro et al., "Why Should I Trust You?: Explaining the Predictions of 
    Any Classifier", https://arxiv.org/abs/1602.04938
"""

import numpy as np
import matplotlib.pyplot as plt
from lime import lime_image
from skimage.segmentation import mark_boundaries

import torch
import torch.nn.functional as F

import sys
sys.path.append('..')
from config import LIME_NUM_SAMPLES, DPI


class LimeExplainer:
    """
    LIME explainer for image classification models.
    
    This class implements LIME, which explains individual predictions by
    approximating the model locally with an interpretable model.
    """
    
    def __init__(self, model, device=None):
        """
        Initialize LimeExplainer.
        
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
        
        # Create LIME explainer
        self.explainer = lime_image.LimeImageExplainer()
    
    def predict_fn(self, images):
        """
        Prediction function to be used by LIME.
        
        Args:
            images (numpy.ndarray): Batch of images of shape (N, H, W, C)
            
        Returns:
            numpy.ndarray: Predicted probabilities of shape (N, num_classes)
        """
        # Convert to PyTorch tensor
        batch = torch.stack([
            self._preprocess_image(img) for img in images
        ])
        
        # Move to device
        batch = batch.to(self.device)
        
        # Get predictions
        with torch.no_grad():
            logits = self.model(batch)
            probs = F.softmax(logits, dim=1)
            
        return probs.cpu().numpy()
    
    def _preprocess_image(self, img_array):
        """
        Preprocess a single image for the model.
        
        Args:
            img_array (numpy.ndarray): Image array of shape (H, W, C)
            
        Returns:
            torch.Tensor: Preprocessed image tensor of shape (C, H, W)
        """
        # The input to this method is already a numpy array
        # We need to convert it to a PyTorch tensor and normalize it
        
        # Convert to float
        img_array = img_array.astype(np.float32)
        
        # Normalize to [0, 1]
        if img_array.max() > 1.0:
            img_array = img_array / 255.0
            
        # Convert to tensor and permute dimensions
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
        
        # Normalize with ImageNet mean and std
        img_tensor = self._normalize(img_tensor)
        
        return img_tensor
    
    def _normalize(self, img_tensor):
        """
        Normalize a tensor with ImageNet mean and std.
        
        Args:
            img_tensor (torch.Tensor): Image tensor of shape (C, H, W)
            
        Returns:
            torch.Tensor: Normalized image tensor
        """
        mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
        
        return (img_tensor - mean) / std
    
    def explain(self, img_array, target_class=None, num_samples=LIME_NUM_SAMPLES,
               num_features=5, positive_only=True, hide_rest=True):
        """
        Generate a LIME explanation for the given image.
        
        Args:
            img_array (numpy.ndarray): Image array of shape (H, W, C)
            target_class (int, optional): Target class index
                If None, uses the predicted class
            num_samples (int): Number of samples to generate around the image
            num_features (int): Maximum number of features (superpixels) to show
            positive_only (bool): Whether to only show positive contributions
            hide_rest (bool): Whether to hide superpixels not in the explanation
            
        Returns:
            tuple: (explanation, visualization)
        """
        # Make sure image is in the right format
        if img_array.max() > 1.0:
            img_array = img_array / 255.0
        
        # If target_class is None, use the predicted class
        if target_class is None:
            # Preprocess image
            img_tensor = self._preprocess_image(img_array)
            img_tensor = img_tensor.unsqueeze(0).to(self.device)
            
            # Get prediction
            with torch.no_grad():
                output = self.model(img_tensor)
                target_class = output.argmax(dim=1).item()
        
        # Generate explanation
        explanation = self.explainer.explain_instance(
            img_array, 
            self.predict_fn,
            top_labels=5,  # Get explanations for top 5 classes
            hide_color=0,
            num_samples=num_samples
        )
        
        # Get visualization for the target class
        temp, mask = explanation.get_image_and_mask(
            target_class, 
            positive_only=positive_only, 
            num_features=num_features, 
            hide_rest=hide_rest
        )
        
        visualization = mark_boundaries(temp, mask)
        
        return explanation, visualization
    
    def visualize(self, img_array, target_class=None, num_samples=LIME_NUM_SAMPLES,
                 num_features=5, positive_only=True, hide_rest=True,
                 figsize=(12, 5), save_path=None):
        """
        Visualize a LIME explanation for the given image.
        
        Args:
            img_array (numpy.ndarray): Image array of shape (H, W, C)
            target_class (int, optional): Target class index
                If None, uses the predicted class
            num_samples (int): Number of samples to generate around the image
            num_features (int): Maximum number of features (superpixels) to show
            positive_only (bool): Whether to only show positive contributions
            hide_rest (bool): Whether to hide superpixels not in the explanation
            figsize (tuple): Figure size
            save_path (str, optional): Path to save the visualization
            
        Returns:
            tuple: (explanation, visualization)
        """
        # Make sure image is in the right format
        if img_array.max() > 1.0:
            img_array = img_array / 255.0
        
        # Generate explanation
        explanation, visualization = self.explain(
            img_array, target_class, num_samples,
            num_features, positive_only, hide_rest
        )
        
        # If target_class is None, it was set in the explain method
        if target_class is None:
            # Preprocess image
            img_tensor = self._preprocess_image(img_array)
            img_tensor = img_tensor.unsqueeze(0).to(self.device)
            
            # Get prediction
            with torch.no_grad():
                output = self.model(img_tensor)
                target_class = output.argmax(dim=1).item()
                confidence = F.softmax(output, dim=1)[0, target_class].item()
        else:
            # Get confidence for the target class
            img_tensor = self._preprocess_image(img_array)
            img_tensor = img_tensor.unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self.model(img_tensor)
                confidence = F.softmax(output, dim=1)[0, target_class].item()
        
        # Plot the visualization
        plt.figure(figsize=figsize)
        
        plt.subplot(1, 2, 1)
        plt.imshow(img_array)
        plt.title(f"Original Image\nClass: {target_class} ({confidence:.2f})")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(visualization)
        plt.title("LIME Explanation\nRegions supporting prediction")
        plt.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
        
        plt.show()
        
        return explanation, visualization


def apply_lime(model, image_array, target_class=None, num_features=5, 
              save_path=None, device=None):
    """
    Apply LIME to an image.
    
    Args:
        model (torch.nn.Module): The model to explain
        image_array (numpy.ndarray): Image array of shape (H, W, C)
        target_class (int, optional): Target class index
            If None, uses the predicted class
        num_features (int): Maximum number of features (superpixels) to show
        save_path (str, optional): Path to save the visualization
        device (torch.device, optional): Device to run the model on
        
    Returns:
        tuple: (explanation, visualization)
    """
    # Create LIME explainer
    explainer = LimeExplainer(model, device)
    
    # Visualize
    explanation, visualization = explainer.visualize(
        image_array,
        target_class=target_class,
        num_features=num_features,
        save_path=save_path
    )
    
    return explanation, visualization