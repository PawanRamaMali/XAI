"""
CNN model architectures for skin cancer classification.

This module defines the neural network architectures used for 
skin lesion classification, leveraging transfer learning from 
pre-trained models.
"""

import torch
import torch.nn as nn
import torchvision.models as models

import sys
sys.path.append('..')
from config import MODEL_NAME, PRETRAINED, NUM_CLASSES


class SkinLesionModel(nn.Module):
    """
    Model for skin lesion classification using transfer learning.
    
    This class creates a model based on a pre-trained CNN architecture
    and adapts it for skin lesion classification.
    """
    
    def __init__(self, model_name=MODEL_NAME, num_classes=NUM_CLASSES, pretrained=PRETRAINED):
        """
        Initialize the model.
        
        Args:
            model_name (str): Name of the pre-trained model to use
            num_classes (int): Number of output classes
            pretrained (bool): Whether to use pre-trained weights
        """
        super(SkinLesionModel, self).__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        
        # Initialize the base model
        if model_name == 'resnet18':
            self.base_model = models.resnet18(pretrained=pretrained)
            num_features = self.base_model.fc.in_features
            self.base_model.fc = nn.Identity()  # Remove classification layer
        
        elif model_name == 'resnet50':
            self.base_model = models.resnet50(pretrained=pretrained)
            num_features = self.base_model.fc.in_features
            self.base_model.fc = nn.Identity()  # Remove classification layer
            
        elif model_name == 'efficientnet_b0':
            self.base_model = models.efficientnet_b0(pretrained=pretrained)
            num_features = self.base_model.classifier[1].in_features
            self.base_model.classifier = nn.Identity()  # Remove classification layer
            
        elif model_name == 'densenet121':
            self.base_model = models.densenet121(pretrained=pretrained)
            num_features = self.base_model.classifier.in_features
            self.base_model.classifier = nn.Identity()  # Remove classification layer
            
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # Add custom classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        
        # Initialize the custom layers
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, H, W)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes)
        """
        # Extract features from the base model
        features = self.base_model(x)
        
        # Apply classifier
        logits = self.classifier(features)
        
        return logits
    
    def freeze_base_model(self):
        """
        Freeze the parameters of the base model to prevent them from updating during training.
        """
        for param in self.base_model.parameters():
            param.requires_grad = False
            
    def unfreeze_base_model(self, layers=None):
        """
        Unfreeze specified layers of the base model for fine-tuning.
        
        Args:
            layers (list, optional): List of layer indices to unfreeze.
                If None, unfreeze all layers.
        """
        if layers is None:
            # Unfreeze all layers
            for param in self.base_model.parameters():
                param.requires_grad = True
        else:
            # For ResNet-like models
            if 'resnet' in self.model_name:
                modules = [
                    self.base_model.conv1,
                    self.base_model.bn1,
                    self.base_model.layer1,
                    self.base_model.layer2,
                    self.base_model.layer3,
                    self.base_model.layer4
                ]
                
                for idx in layers:
                    if idx < len(modules):
                        for param in modules[idx].parameters():
                            param.requires_grad = True
            
            # For EfficientNet
            elif 'efficientnet' in self.model_name:
                modules = list(self.base_model.features)
                
                for idx in layers:
                    if idx < len(modules):
                        for param in modules[idx].parameters():
                            param.requires_grad = True
            
            # For DenseNet
            elif 'densenet' in self.model_name:
                modules = [
                    self.base_model.features.conv0,
                    self.base_model.features.norm0,
                    self.base_model.features.denseblock1,
                    self.base_model.features.denseblock2,
                    self.base_model.features.denseblock3,
                    self.base_model.features.denseblock4
                ]
                
                for idx in layers:
                    if idx < len(modules):
                        for param in modules[idx].parameters():
                            param.requires_grad = True
    
    def get_target_layer(self):
        """
        Get the target layer for Grad-CAM.
        
        Returns:
            torch.nn.Module: The target layer
        """
        # For ResNet models
        if 'resnet' in self.model_name:
            return self.base_model.layer4[-1]
        
        # For EfficientNet
        elif 'efficientnet' in self.model_name:
            return self.base_model.features[-1]
        
        # For DenseNet
        elif 'densenet' in self.model_name:
            return self.base_model.features.denseblock4
        
        else:
            raise ValueError(f"Unsupported model for Grad-CAM: {self.model_name}")


def create_model(model_name=MODEL_NAME, num_classes=NUM_CLASSES, pretrained=PRETRAINED):
    """
    Factory function to create a model instance.
    
    Args:
        model_name (str): Name of the pre-trained model to use
        num_classes (int): Number of output classes
        pretrained (bool): Whether to use pre-trained weights
        
    Returns:
        SkinLesionModel: Instantiated model
    """
    model = SkinLesionModel(model_name, num_classes, pretrained)
    
    # Initially freeze the base model for transfer learning
    model.freeze_base_model()
    
    return model


def load_model(model_path, model_name=MODEL_NAME, num_classes=NUM_CLASSES):
    """
    Load a saved model from disk.
    
    Args:
        model_path (str): Path to the saved model
        model_name (str): Name of the pre-trained model used
        num_classes (int): Number of output classes
        
    Returns:
        SkinLesionModel: Loaded model
    """
    model = create_model(model_name, num_classes, pretrained=False)
    
    # Load the state dict
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    
    return model


def print_model_summary(model):
    """
    Print a summary of the model architecture.
    
    Args:
        model (torch.nn.Module): The model to summarize
    """
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model Architecture: {model.model_name}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    
    # Print layers
    print("\nModel Layers:")
    for name, module in model.named_children():
        print(f"- {name}:")
        if name == 'base_model':
            # For base model, just print a summary
            if 'resnet' in model.model_name:
                print(f"  - ResNet architecture with {len(list(module.children()))} main blocks")
            elif 'efficientnet' in model.model_name:
                print(f"  - EfficientNet architecture with {len(list(module.features))}" 
                      f" feature blocks")
            elif 'densenet' in model.model_name:
                print(f"  - DenseNet architecture with {len(list(module.features))}" 
                     f" feature blocks")
        else:
            # For custom layers, print details
            for idx, layer in enumerate(module):
                print(f"  - {idx}: {layer}")