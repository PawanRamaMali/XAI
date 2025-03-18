"""
Configuration parameters for the skin cancer detection and explainability project.

This file contains all configurable parameters to allow for easy experimentation
without modifying core code.
"""

import os
from pathlib import Path

# Project paths
# Update these paths to match your system
PROJECT_ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = PROJECT_ROOT / "data"
METADATA_PATH = DATA_DIR / "HAM10000_metadata.csv"
IMAGES_DIR = DATA_DIR / "HAM10000_images"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = OUTPUT_DIR / "models"
RESULTS_DIR = OUTPUT_DIR / "results"

# Create directories if they don't exist
for directory in [DATA_DIR, OUTPUT_DIR, MODEL_DIR, RESULTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Example image path for demonstrations
EXAMPLE_IMG_PATH = DATA_DIR / "example_melanoma.jpg"

# Model parameters
MODEL_NAME = "resnet50"  # Options: "resnet50", "efficientnet_b0", "densenet121"
PRETRAINED = True
NUM_CLASSES = 7  # HAM10000 has 7 classes
MODEL_PATH = MODEL_DIR / "skin_cancer_model.pth"

# Training parameters
BATCH_SIZE = 32
NUM_EPOCHS = 15
LEARNING_RATE = 0.001
SCHEDULER_STEP_SIZE = 7
SCHEDULER_GAMMA = 0.1
EARLY_STOPPING_PATIENCE = 5
WEIGHT_DECAY = 1e-4
RANDOM_SEED = 42
NUM_WORKERS = 4  # Number of workers for data loading

# Data parameters
IMAGE_SIZE = 224  # Input image size (224x224 pixels)
TRAIN_VAL_SPLIT = 0.8  # 80% training, 20% validation
DATA_AUGMENTATION = True  # Whether to use data augmentation

# Explain parameters
GRADCAM_LAYER = "layer4"  # Layer to use for GradCAM (for ResNet)
IG_STEPS = 50  # Steps for Integrated Gradients
LIME_NUM_SAMPLES = 1000  # Number of samples for LIME
SHAP_NUM_SAMPLES = 100  # Number of background samples for SHAP

# Visualization parameters
COLORMAP = "jet"  # Colormap for heatmaps
OVERLAY_ALPHA = 0.4  # Transparency for overlay visualizations
DPI = 100  # DPI for saved figures