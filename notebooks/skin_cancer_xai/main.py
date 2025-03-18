"""
Main execution script for skin cancer classification and explanation.

This script provides a comprehensive demonstration of the skin cancer
classification model, including data exploration, model training,
evaluation, and explainability visualizations.
"""

import os
import argparse
import torch

# Import project modules
from config import (
    METADATA_PATH, IMAGES_DIR, OUTPUT_DIR, MODEL_PATH, MODEL_NAME,
    NUM_CLASSES, NUM_EPOCHS, IMAGE_SIZE, EXAMPLE_IMG_PATH, RANDOM_SEED
)
from data.dataset import get_data_loaders, load_single_image
from data.preprocessing import explore_dataset, visualize_augmentations
from models.cnn_model import create_model, load_model, print_model_summary
from training.train import train_model, fine_tune_model, save_training_history
from training.evaluate import evaluate_and_save_results
from explanation.gradcam import apply_gradcam
from explanation.lime_explainer import apply_lime
from explanation.integrated_gradients import apply_integrated_gradients
from explanation.shap_explainer import apply_shap
from explanation.cam import apply_cam
from visualization.clinical_report import generate_clinical_report, compare_explanations


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Skin Cancer Classification and XAI Demo')
    
    parser.add_argument('--mode', type=str, default='all',
                        choices=['explore', 'train', 'evaluate', 'explain', 'all'],
                        help='Mode of operation')
    
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Path to the dataset directory')
    
    parser.add_argument('--metadata', type=str, default=None,
                        help='Path to the metadata CSV file')
    
    parser.add_argument('--images_dir', type=str, default=None,
                        help='Path to the images directory')
    
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to save or load the model')
    
    parser.add_argument('--model_name', type=str, default=MODEL_NAME,
                        choices=['resnet18', 'resnet50', 'efficientnet_b0', 'densenet121'],
                        help='Name of the model architecture')
    
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS,
                        help='Number of training epochs')
    
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (e.g., "cuda:0", "cpu")')
    
    parser.add_argument('--seed', type=int, default=RANDOM_SEED,
                        help='Random seed for reproducibility')
    
    parser.add_argument('--example_img', type=str, default=None,
                        help='Path to an example image for explanation')
    
    return parser.parse_args()


def main():
    """Main execution function."""
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set paths
    metadata_path = args.metadata if args.metadata else METADATA_PATH
    images_dir = args.images_dir if args.images_dir else IMAGES_DIR
    model_path = args.model_path if args.model_path else MODEL_PATH
    example_img_path = args.example_img if args.example_img else EXAMPLE_IMG_PATH
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Set mode
    if args.mode == 'all':
        modes = ['explore', 'train', 'evaluate', 'explain']
    else:
        modes = [args.mode]
    
    # Execute modes
    if 'explore' in modes:
        print("\n" + "="*80)
        print("DATA EXPLORATION")
        print("="*80)
        
        # Explore dataset
        df = explore_dataset(metadata_path, images_dir)
        
        # Visualize data augmentations
        visualize_augmentations(metadata_path, images_dir)
    
    # Get data loaders (needed for training and evaluation)
    if set(modes) & set(['train', 'evaluate']):
        train_loader, val_loader, class_to_idx = get_data_loaders(
            metadata_path, images_dir
        )
        
        # Create a list of class names in correct order
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    
    if 'train' in modes:
        print("\n" + "="*80)
        print("MODEL TRAINING")
        print("="*80)
        
        # Create model
        model = create_model(args.model_name, len(class_to_idx))
        
        # Print model summary
        print_model_summary(model)
        
        # Train model
        trained_model, history = train_model(
            model, train_loader, val_loader, 
            num_epochs=args.epochs, device=device
        )
        
        # Save training history
        save_training_history(history)
        
        print("\nFine-tuning the model...")
        
        # Fine-tune model
        fine_tuned_model, ft_history = fine_tune_model(
            trained_model, train_loader, val_loader,
            layers_to_unfreeze=[-20, -15, -10, -5],  # Unfreeze last few layers
            device=device
        )
        
        # Save fine-tuning history
        save_training_history(ft_history, filename=os.path.join(OUTPUT_DIR, 'fine_tuning_history.csv'))
        
        # Use fine-tuned model for subsequent steps
        model = fine_tuned_model
        
    elif set(modes) & set(['evaluate', 'explain']):
        # Load model if not training
        print(f"Loading model from {model_path}...")
        if os.path.exists(model_path):
            model = load_model(model_path, args.model_name, NUM_CLASSES)
            print("Model loaded successfully.")
        else:
            print(f"Model file not found at {model_path}. Creating a new model.")
            model = create_model(args.model_name, NUM_CLASSES)
    
    if 'evaluate' in modes:
        print("\n" + "="*80)
        print("MODEL EVALUATION")
        print("="*80)
        
        # Evaluate model
        metrics = evaluate_and_save_results(
            model, val_loader, class_names, device=device
        )
        
        print(f"Overall accuracy: {metrics['accuracy']:.4f}")
    
    if 'explain' in modes:
        print("\n" + "="*80)
        print("MODEL EXPLAINABILITY")
        print("="*80)
        
        # Load example image
        if not os.path.exists(example_img_path):
            print(f"Example image not found at {example_img_path}. Using a sample from the validation set.")
            # Get a sample image from the validation set
            for batch in val_loader:
                inputs, labels = batch
                example_img_tensor = inputs[0].unsqueeze(0)
                example_class = labels[0].item()
                # Convert tensor to numpy image for visualization
                from data.dataset import get_inverse_transform
                import numpy as np
                inverse_transform = get_inverse_transform()
                example_img_np = inverse_transform(example_img_tensor[0]).permute(1, 2, 0).cpu().numpy()
                example_img_np = np.clip(example_img_np, 0, 1)
                break
        else:
            # Load provided example image
            original_img, example_img_tensor = load_single_image(example_img_path)
            example_img_np = np.array(original_img) / 255.0
            
            # Get prediction
            with torch.no_grad():
                outputs = model(example_img_tensor.to(device))
                example_class = outputs.argmax(dim=1).item()
        
        print(f"Using image with predicted class: {class_names[example_class]}")
        
        # Apply Grad-CAM
        print("\nApplying Grad-CAM...")
        gradcam_heatmap, gradcam_overlay = apply_gradcam(
            model, example_img_tensor, example_img_np, 
            target_class=example_class, device=device
        )
        
        # Apply LIME
        print("\nApplying LIME...")
        lime_exp, lime_viz = apply_lime(
            model, example_img_np,
            target_class=example_class, device=device
        )
        
        # Apply Integrated Gradients
        print("\nApplying Integrated Gradients...")
        ig_attrs = apply_integrated_gradients(
            model, example_img_tensor, example_img_np,
            target_class=example_class, device=device
        )
        
        # Apply SHAP (skip for large images to avoid memory issues)
        if example_img_tensor.shape[2] * example_img_tensor.shape[3] < 150*150:
            print("\nApplying SHAP...")
            shap_values = apply_shap(
                model, example_img_tensor, example_img_np,
                target_class=example_class, device=device
            )
        else:
            print("\nSkipping SHAP for large images to avoid memory issues.")
        
        # Apply CAM
        print("\nApplying Class Activation Mapping...")
        try:
            cam_heatmap, cam_overlay = apply_cam(
                model, example_img_tensor, example_img_np,
                target_class=example_class, device=device
            )
        except Exception as e:
            print(f"Error applying CAM: {e}")
        
        # Compare all explanation methods
        print("\nComparing all explanation methods...")
        compare_explanations(
            model, example_img_tensor, example_img_np,
            target_class=example_class, device=device,
            save_path=os.path.join(OUTPUT_DIR, "explanation_comparison.png")
        )
        
        # Generate clinical report
        print("\nGenerating clinical report...")
        report_data = generate_clinical_report(
            model, example_img_tensor, example_img_np,
            original_path=example_img_path if os.path.exists(example_img_path) else "sample_image.jpg",
            class_names=class_names, device=device
        )
        
        print(f"Clinical report generated and saved to: {report_data['comprehensive_path']}")
    
    print("\n" + "="*80)
    print("DEMONSTRATION COMPLETED")
    print("="*80)
    

if __name__ == "__main__":
    main()