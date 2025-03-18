"""
Clinical reporting functionality for skin cancer classification.

This module provides functions for generating clinical reports and
comparing different explainability methods.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn.functional as F

import sys
sys.path.append('..')

from explanation.gradcam import apply_gradcam
from explanation.lime_explainer import apply_lime
from explanation.integrated_gradients import apply_integrated_gradients
from explanation.shap_explainer import apply_shap
from explanation.cam import apply_cam


def compare_explanations(model, image_tensor, original_image, target_class=None, 
                         device=None, save_path=None):
    """
    Compare different explanation methods on the same image.
    
    Args:
        model (torch.nn.Module): The model to explain
        image_tensor (torch.Tensor): Input image tensor of shape (1, C, H, W)
        original_image (numpy.ndarray): Original image array of shape (H, W, 3)
        target_class (int, optional): Target class index
            If None, uses the predicted class
        device (torch.device, optional): Device to run the model on
        save_path (str, optional): Path to save the comparison visualization
        
    Returns:
        dict: Dictionary containing the results of each explanation method
    """
    # Use GPU if available
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Move model and tensor to device
    model = model.to(device)
    image_tensor = image_tensor.to(device)
    
    # Get the predicted class if target_class is None
    if target_class is None:
        with torch.no_grad():
            output = model(image_tensor)
            target_class = output.argmax(dim=1).item()
            confidence = F.softmax(output, dim=1)[0, target_class].item()
    else:
        with torch.no_grad():
            output = model(image_tensor)
            confidence = F.softmax(output, dim=1)[0, target_class].item()
    
    # Apply different explanation methods
    print("Applying Grad-CAM...")
    gradcam_heatmap, gradcam_overlay = apply_gradcam(
        model, image_tensor, original_image, 
        target_class=target_class, device=device
    )
    
    print("Applying LIME...")
    lime_exp, lime_viz = apply_lime(
        model, original_image,
        target_class=target_class, device=device
    )
    
    print("Applying Integrated Gradients...")
    ig_attrs = apply_integrated_gradients(
        model, image_tensor, original_image,
        target_class=target_class, device=device
    )
    
    # Apply CAM (may fail for some models)
    try:
        print("Applying CAM...")
        cam_heatmap, cam_overlay = apply_cam(
            model, image_tensor, original_image,
            target_class=target_class, device=device
        )
    except Exception as e:
        print(f"Could not apply CAM: {e}")
        cam_heatmap, cam_overlay = None, None
    
    # Create comparison visualization
    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    
    # Original image
    axes[0].imshow(original_image)
    axes[0].set_title(f"Original\nConfidence: {confidence:.2f}")
    axes[0].axis('off')
    
    # Grad-CAM
    axes[1].imshow(gradcam_overlay)
    axes[1].set_title("Grad-CAM")
    axes[1].axis('off')
    
    # LIME
    axes[2].imshow(lime_viz)
    axes[2].set_title("LIME")
    axes[2].axis('off')
    
    # Integrated Gradients
    # Create a visualization for IG (blending with original)
    ig_vis = np.zeros_like(original_image)
    ig_vis[:,:,0] = ig_attrs  # Use attribution as red channel
    ig_vis = 0.7 * original_image + 0.3 * ig_vis
    
    axes[3].imshow(ig_vis)
    axes[3].set_title("Integrated Gradients")
    axes[3].axis('off')
    
    # CAM
    if cam_overlay is not None:
        axes[4].imshow(cam_overlay)
        axes[4].set_title("CAM")
    else:
        axes[4].imshow(np.zeros_like(original_image))
        axes[4].set_title("CAM (Failed)")
    axes[4].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
    
    plt.show()
    
    # Return results
    results = {
        'gradcam': (gradcam_heatmap, gradcam_overlay),
        'lime': (lime_exp, lime_viz),
        'integrated_gradients': ig_attrs,
        'cam': (cam_heatmap, cam_overlay) if cam_heatmap is not None else None
    }
    
    return results


def generate_clinical_report(model, image_tensor, original_image, original_path=None,
                            class_names=None, device=None):
    """
    Generate a comprehensive clinical report with explanations.
    
    Args:
        model (torch.nn.Module): The model to explain
        image_tensor (torch.Tensor): Input image tensor of shape (1, C, H, W)
        original_image (numpy.ndarray): Original image array of shape (H, W, 3)
        original_path (str, optional): Path to the original image file
        class_names (list, optional): List of class names
        device (torch.device, optional): Device to run the model on
        
    Returns:
        dict: Dictionary containing report information and paths
    """
    # Use GPU if available
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Move model and tensor to device
    model = model.to(device)
    image_tensor = image_tensor.to(device)
    
    # Create output directory
    report_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                              "outputs", "reports")
    os.makedirs(report_dir, exist_ok=True)
    
    # Generate a filename based on original path or timestamp
    if original_path:
        basename = os.path.splitext(os.path.basename(original_path))[0]
    else:
        from datetime import datetime
        basename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Get model prediction
    with torch.no_grad():
        output = model(image_tensor.to(device))
        probabilities = F.softmax(output, dim=1)[0].cpu().numpy()
        predicted_class = output.argmax(dim=1).item()
        confidence = probabilities[predicted_class]
    
    # Generate class names if not provided
    if class_names is None:
        class_names = [f"Class {i}" for i in range(len(probabilities))]
    
    # Get explanations
    explanations = compare_explanations(
        model, image_tensor, original_image, target_class=predicted_class,
        device=device, save_path=os.path.join(report_dir, f"{basename}_explanations.png")
    )
    
    # Save predicted probabilities
    plt.figure(figsize=(10, 6))
    y_pos = np.arange(len(class_names))
    plt.barh(y_pos, probabilities)
    plt.yticks(y_pos, class_names)
    plt.xlabel('Probability')
    plt.title('Prediction Probabilities')
    plt.tight_layout()
    plt.savefig(os.path.join(report_dir, f"{basename}_probabilities.png"), bbox_inches='tight')
    
    # Create a comprehensive report
    diagnosis = class_names[predicted_class]
    
    # Generate simple HTML report
    html_path = os.path.join(report_dir, f"{basename}_report.html")
    with open(html_path, 'w') as f:
        f.write(f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Clinical Report - {diagnosis}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .container {{ max-width: 1000px; margin: 0 auto; }}
                .header {{ text-align: center; margin-bottom: 20px; }}
                .image-row {{ display: flex; justify-content: center; margin-bottom: 20px; }}
                .image-container {{ margin: 10px; text-align: center; }}
                img {{ max-width: 100%; }}
                table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; }}
                th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
                .footer {{ text-align: center; margin-top: 30px; font-size: 0.8em; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Skin Lesion Analysis Report</h1>
                    <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
                
                <h2>Patient Information</h2>
                <table>
                    <tr>
                        <th>Image ID:</th>
                        <td>{basename}</td>
                    </tr>
                    <tr>
                        <th>Diagnosis:</th>
                        <td>{diagnosis}</td>
                    </tr>
                    <tr>
                        <th>Confidence:</th>
                        <td>{confidence:.2f}</td>
                    </tr>
                </table>
                
                <h2>Original Image</h2>
                <div class="image-row">
                    <div class="image-container">
                        <img src="{basename}_original.png" alt="Original Image">
                        <p>Original Image</p>
                    </div>
                </div>
                
                <h2>Probability Distribution</h2>
                <div class="image-row">
                    <div class="image-container">
                        <img src="{basename}_probabilities.png" alt="Probability Distribution">
                        <p>Probability of each diagnosis</p>
                    </div>
                </div>
                
                <h2>Model Explanations</h2>
                <div class="image-row">
                    <div class="image-container">
                        <img src="{basename}_explanations.png" alt="Model Explanations">
                        <p>Different explainability methods showing regions of interest</p>
                    </div>
                </div>
                
                <div class="footer">
                    <p>This report was generated using AI assistance and should be reviewed by a medical professional.</p>
                </div>
            </div>
        </body>
        </html>
        """)
    
    # Save original image
    plt.figure(figsize=(5, 5))
    plt.imshow(original_image)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(report_dir, f"{basename}_original.png"), bbox_inches='tight')
    
    print(f"Clinical report generated at: {html_path}")
    
    return {
        'diagnosis': diagnosis,
        'confidence': confidence,
        'probabilities': probabilities,
        'class_names': class_names,
        'explanations': explanations,
        'original_path': original_path,
        'report_dir': report_dir,
        'comprehensive_path': html_path
    }