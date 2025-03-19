"""
Updated clinical_report.py to use Captum-based Grad-CAM implementation.

This module integrates the Captum-based Grad-CAM implementation into the
clinical report generation pipeline for better explainability.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec
import cv2
from datetime import datetime

import torch
import torch.nn.functional as F

import sys
sys.path.append('..')
from config import OUTPUT_DIR, DPI, COLORMAP, OVERLAY_ALPHA

# Import the Captum-based Grad-CAM implementation
from explanation.captum_gradcam import apply_captum_gradcam
from explanation.lime_explainer import apply_lime
from explanation.integrated_gradients import apply_integrated_gradients
from explanation.shap_explainer import apply_shap
from explanation.cam import apply_cam


def compare_explanations(model, image_tensor, original_image, target_class=None, 
                        device=None, save_path=None):
    """
    Compare different explanation methods side by side.
    
    Args:
        model (torch.nn.Module): The model to explain
        image_tensor (torch.Tensor): Input image tensor of shape (1, C, H, W)
        original_image (numpy.ndarray): Original image array of shape (H, W, 3)
        target_class (int, optional): Target class index
            If None, uses the predicted class
        device (torch.device, optional): Device to run the model on
        save_path (str, optional): Path to save the visualization
    """
    # Get the predicted class and confidence
    if device is None:
        device = next(model.parameters()).device
    
    # Send model to device
    model.to(device)
    
    # Get prediction
    if target_class is None:
        with torch.no_grad():
            output = model(image_tensor.to(device))
            target_class = output.argmax(dim=1).item()
            confidence = F.softmax(output, dim=1)[0, target_class].item()
    else:
        with torch.no_grad():
            output = model(image_tensor.to(device))
            confidence = F.softmax(output, dim=1)[0, target_class].item()
    
    # Create a figure
    plt.figure(figsize=(20, 10))
    
    # Original image
    plt.subplot(2, 3, 1)
    plt.imshow(original_image)
    plt.title(f"Original Image\nClass: {target_class} ({confidence:.2f})")
    plt.axis('off')
    
    # Try Captum Grad-CAM (our new implementation)
    try:
        print("Applying Captum Grad-CAM for comparison...")
        gradcam_heatmap, gradcam_overlay = apply_captum_gradcam(
            model, image_tensor, original_image, 
            target_class=target_class, device=device
        )
        
        plt.subplot(2, 3, 2)
        plt.imshow(gradcam_overlay)
        plt.title("Captum Grad-CAM")
        plt.axis('off')
    except Exception as e:
        print(f"Error applying Captum Grad-CAM for comparison: {e}")
        plt.subplot(2, 3, 2)
        plt.text(0.5, 0.5, "Grad-CAM not available", 
                 ha='center', va='center', transform=plt.gca().transAxes)
        plt.axis('off')
    
    # Try LIME
    try:
        print("Applying LIME for comparison...")
        lime_exp, lime_viz = apply_lime(
            model, original_image, 
            target_class=target_class, device=device
        )
        
        plt.subplot(2, 3, 3)
        plt.imshow(lime_viz)
        plt.title("LIME")
        plt.axis('off')
    except Exception as e:
        print(f"Error applying LIME for comparison: {e}")
        plt.subplot(2, 3, 3)
        plt.text(0.5, 0.5, "LIME not available", 
                 ha='center', va='center', transform=plt.gca().transAxes)
        plt.axis('off')
    
    # Try Integrated Gradients
    try:
        print("Applying Integrated Gradients for comparison...")
        ig_attrs = apply_integrated_gradients(
            model, image_tensor, original_image,
            target_class=target_class, device=device
        )
        
        # Create a visualization using the attribution map
        cmap = plt.get_cmap(COLORMAP)
        ig_heatmap = cmap(ig_attrs)[:, :, :3]
        
        # Resize to match original image
        ig_heatmap_resized = cv2.resize(
            ig_heatmap, (original_image.shape[1], original_image.shape[0])
        )
        
        # Create overlay
        ig_overlay = (1 - OVERLAY_ALPHA) * original_image + OVERLAY_ALPHA * ig_heatmap_resized
        ig_overlay = np.clip(ig_overlay, 0, 1)
        
        plt.subplot(2, 3, 4)
        plt.imshow(ig_overlay)
        plt.title("Integrated Gradients")
        plt.axis('off')
    except Exception as e:
        print(f"Error applying Integrated Gradients for comparison: {e}")
        plt.subplot(2, 3, 4)
        plt.text(0.5, 0.5, "Integrated Gradients not available", 
                 ha='center', va='center', transform=plt.gca().transAxes)
        plt.axis('off')
    
    # Try CAM
    try:
        print("Applying CAM for comparison...")
        cam_heatmap, cam_overlay = apply_cam(
            model, image_tensor, original_image,
            target_class=target_class, device=device
        )
        
        plt.subplot(2, 3, 5)
        plt.imshow(cam_overlay)
        plt.title("Class Activation Mapping")
        plt.axis('off')
    except Exception as e:
        print(f"Error applying CAM for comparison: {e}")
        plt.subplot(2, 3, 5)
        plt.text(0.5, 0.5, "CAM not available", 
                 ha='center', va='center', transform=plt.gca().transAxes)
        plt.axis('off')
    
    # Try SHAP (if image is small enough)
    if image_tensor.shape[2] * image_tensor.shape[3] < 150*150:
        try:
            print("Applying SHAP for comparison...")
            shap_values = apply_shap(
                model, image_tensor, original_image,
                target_class=target_class, device=device
            )
            
            # Use the SHAP values to create a visualization
            cmap = plt.get_cmap(COLORMAP)
            shap_heatmap = np.abs(shap_values).sum(axis=1)[0]
            shap_heatmap = shap_heatmap / np.max(shap_heatmap)
            shap_heatmap_rgb = cmap(shap_heatmap)[:, :, :3]
            
            # Resize to match original image
            shap_heatmap_resized = cv2.resize(
                shap_heatmap_rgb, (original_image.shape[1], original_image.shape[0])
            )
            
            # Create overlay
            shap_overlay = (1 - OVERLAY_ALPHA) * original_image + OVERLAY_ALPHA * shap_heatmap_resized
            shap_overlay = np.clip(shap_overlay, 0, 1)
            
            plt.subplot(2, 3, 6)
            plt.imshow(shap_overlay)
            plt.title("SHAP")
            plt.axis('off')
        except Exception as e:
            print(f"Error applying SHAP for comparison: {e}")
            plt.subplot(2, 3, 6)
            plt.text(0.5, 0.5, "SHAP not available", 
                     ha='center', va='center', transform=plt.gca().transAxes)
            plt.axis('off')
    else:
        plt.subplot(2, 3, 6)
        plt.text(0.5, 0.5, "SHAP not available\n(image too large)", 
                 ha='center', va='center', transform=plt.gca().transAxes)
        plt.axis('off')
    
    plt.suptitle("Comparison of Explanation Methods", fontsize=16)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=DPI, bbox_inches='tight')
    
    plt.show()


def generate_clinical_report(model, image_tensor, original_image, original_path=None,
                          class_names=None, device=None, output_dir=OUTPUT_DIR,
                          report_name=None):
    """
    Generate a comprehensive clinical report with prediction and explanations.
    
    Args:
        model (torch.nn.Module): The model to explain
        image_tensor (torch.Tensor): Input image tensor of shape (1, C, H, W)
        original_image (numpy.ndarray): Original image array of shape (H, W, 3)
        original_path (str, optional): Path to the original image
        class_names (list, optional): List of class names
        device (torch.device, optional): Device to run the model on
        output_dir (str): Directory to save the report
        report_name (str, optional): Name of the report file
        
    Returns:
        dict: Dictionary with report paths and information
    """
    # Get the predicted class and confidence
    if device is None:
        device = next(model.parameters()).device
    
    # Send model to device
    model.to(device)
    
    # Get prediction
    with torch.no_grad():
        output = model(image_tensor.to(device))
        probs = F.softmax(output, dim=1)
    
    # Get top-3 predictions
    top_probs, top_classes = torch.topk(probs, k=min(3, probs.size(1)))
    top_probs = top_probs[0].cpu().numpy()
    top_classes = top_classes[0].cpu().numpy()
    
    # Create class names if not provided
    if class_names is None:
        class_names = [f"Class {i}" for i in range(probs.size(1))]
    
    # Generate filename for the report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_name = os.path.basename(original_path) if original_path else "image"
    image_name = os.path.splitext(image_name)[0]
    
    if report_name is None:
        report_name = f"{image_name}_report_{timestamp}"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create PDF paths
    summary_path = os.path.join(output_dir, f"{report_name}_summary.pdf")
    comprehensive_path = os.path.join(output_dir, f"{report_name}_comprehensive.pdf")
    
    # Get main prediction
    pred_class = top_classes[0]
    pred_prob = top_probs[0]
    pred_name = class_names[pred_class]
    
    # Generate explanations
    print("Generating explanations for the report...")
    
    # Create summary PDF
    with PdfPages(summary_path) as pdf:
        # Summary page
        plt.figure(figsize=(8.5, 11))
        plt.suptitle("Skin Lesion Classification Report (Summary)", fontsize=16, y=0.98)
        
        # Add timestamp and metadata
        plt.figtext(0.1, 0.94, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", fontsize=10)
        plt.figtext(0.1, 0.92, f"Image: {image_name}", fontsize=10)
        plt.figtext(0.1, 0.90, f"Model: {model.__class__.__name__}", fontsize=10)
        
        # Create grid for layout
        gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1.5, 1])
        
        # Original image
        ax1 = plt.subplot(gs[0, 0])
        ax1.imshow(original_image)
        ax1.set_title("Original Image")
        ax1.axis('off')
        
        # Prediction results
        ax2 = plt.subplot(gs[0, 1])
        ax2.barh(
            [class_names[idx] for idx in top_classes],
            top_probs,
            color='skyblue'
        )
        ax2.set_title("Prediction Confidence")
        ax2.set_xlim(0, 1)
        
        # Add values to bars
        for i, prob in enumerate(top_probs):
            ax2.text(prob + 0.01, i, f"{prob:.2f}", va='center')
        
        # Try to get Captum Grad-CAM explanation
        try:
            gradcam_heatmap, gradcam_overlay = apply_captum_gradcam(
                model, image_tensor, original_image, 
                target_class=pred_class, device=device
            )
            
            ax3 = plt.subplot(gs[1, 0])
            ax3.imshow(gradcam_overlay)
            ax3.set_title("Grad-CAM Explanation")
            ax3.axis('off')
        except Exception as e:
            print(f"Error applying Captum Grad-CAM: {e}")
            ax3 = plt.subplot(gs[1, 0])
            ax3.text(0.5, 0.5, "Grad-CAM explanation\nnot available", 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.axis('off')
        
        # Try to get LIME explanation
        try:
            lime_exp, lime_viz = apply_lime(
                model, original_image, 
                target_class=pred_class, device=device
            )
            
            ax4 = plt.subplot(gs[1, 1])
            ax4.imshow(lime_viz)
            ax4.set_title("LIME Explanation")
            ax4.axis('off')
        except Exception as e:
            print(f"Error applying LIME: {e}")
            ax4 = plt.subplot(gs[1, 1])
            ax4.text(0.5, 0.5, "LIME explanation\nnot available", 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.axis('off')
        
        # Add conclusion and recommendations
        ax5 = plt.subplot(gs[2, :])
        ax5.axis('off')
        
        conclusion_text = (
            f"Prediction: {pred_name} (Confidence: {pred_prob:.2f})\n\n"
            f"This report was generated automatically by an AI system. "
            f"The highlighted regions in the visualizations indicate areas "
            f"that influenced the model's prediction.\n\n"
            f"IMPORTANT: This is a research prototype and should not be used "
            f"for clinical decision making. Please consult a dermatologist "
            f"for proper diagnosis."
        )
        
        ax5.text(0.01, 0.99, "Conclusion:", fontsize=12, weight='bold',
                va='top', ha='left', transform=ax5.transAxes)
        ax5.text(0.01, 0.90, conclusion_text, fontsize=10, va='top', ha='left',
                transform=ax5.transAxes, wrap=True)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for the suptitle
        pdf.savefig()
        plt.close()
    
    # Create comprehensive PDF
    with PdfPages(comprehensive_path) as pdf:
        # Title page
        plt.figure(figsize=(8.5, 11))
        plt.suptitle("Comprehensive Skin Lesion Analysis Report", fontsize=20, y=0.5)
        plt.figtext(0.5, 0.4, f"Prediction: {pred_name}", fontsize=16, ha='center')
        plt.figtext(0.5, 0.35, f"Confidence: {pred_prob:.2f}", fontsize=14, ha='center')
        plt.figtext(0.5, 0.25, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", fontsize=12, ha='center')
        plt.figtext(0.5, 0.15, "This report is for research purposes only.", fontsize=10, ha='center')
        plt.axis('off')
        pdf.savefig()
        plt.close()
        
        # Image and predictions page
        plt.figure(figsize=(8.5, 11))
        plt.suptitle("Image and Predictions", fontsize=16, y=0.98)
        
        # Original image
        plt.subplot(2, 2, 1)
        plt.imshow(original_image)
        plt.title("Original Image")
        plt.axis('off')
        
        # Prediction results
        plt.subplot(2, 2, 2)
        plt.barh(
            [class_names[idx] for idx in range(len(class_names))],
            probs[0].cpu().numpy(),
            color='skyblue'
        )
        plt.title("Full Prediction Confidence")
        plt.xlim(0, 1)
        
        # Add metadata
        plt.subplot(2, 1, 2)
        plt.axis('off')
        
        metadata_text = (
            f"Image Name: {image_name}\n"
            f"Date Analyzed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Model: {model.__class__.__name__}\n"
            f"Image Size: {original_image.shape[0]}x{original_image.shape[1]}\n"
            f"Top Predictions:\n"
        )
        
        for i, (cls, prob) in enumerate(zip(top_classes, top_probs)):
            metadata_text += f"  {i+1}. {class_names[cls]}: {prob:.4f}\n"
        
        plt.text(0.1, 0.9, metadata_text, fontsize=12, va='top', ha='left', transform=plt.gca().transAxes)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig()
        plt.close()
        
        # Explanations page - Grad-CAM and LIME
        plt.figure(figsize=(8.5, 11))
        plt.suptitle("Explanation Methods - Part 1", fontsize=16, y=0.98)
        
        # Try Captum Grad-CAM
        try:
            gradcam_heatmap, gradcam_overlay = apply_captum_gradcam(
                model, image_tensor, original_image, 
                target_class=pred_class, device=device
            )
            
            plt.subplot(2, 2, 1)
            plt.imshow(gradcam_overlay)
            plt.title("Captum Grad-CAM Explanation")
            plt.axis('off')
            
            plt.subplot(2, 2, 2)
            plt.imshow(gradcam_heatmap, cmap=COLORMAP)
            plt.title("Grad-CAM Heatmap")
            plt.axis('off')
            
            gradcam_description = (
                "Grad-CAM (Gradient-weighted Class Activation Mapping) uses the "
                "gradients flowing into the final convolutional layer to highlight "
                "important regions in the image for the predicted class. Brighter "
                "areas indicate regions that strongly influenced the prediction."
            )
            
            plt.figtext(0.1, 0.55, "Grad-CAM Explanation:", fontsize=12, weight='bold')
            plt.figtext(0.1, 0.5, gradcam_description, fontsize=10, wrap=True)
            
        except Exception as e:
            print(f"Error applying Grad-CAM for report: {e}")
            plt.subplot(2, 2, 1)
            plt.text(0.5, 0.5, "Grad-CAM explanation\nnot available", 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.axis('off')
            
            plt.subplot(2, 2, 2)
            plt.axis('off')
        
        # Try LIME
        try:
            lime_exp, lime_viz = apply_lime(
                model, original_image, 
                target_class=pred_class, device=device
            )
            
            plt.subplot(2, 2, 3)
            plt.imshow(lime_viz)
            plt.title("LIME Explanation")
            plt.axis('off')
            
            lime_description = (
                "LIME (Local Interpretable Model-agnostic Explanations) perturbs the "
                "input image by segmenting it and turning segments on or off. "
                "It then fits a simple model to approximate how the segments "
                "affect the model's prediction. Green regions positively "
                "contribute to the prediction, while red regions negatively contribute."
            )
            
            plt.figtext(0.1, 0.25, "LIME Explanation:", fontsize=12, weight='bold')
            plt.figtext(0.1, 0.2, lime_description, fontsize=10, wrap=True)
            
        except Exception as e:
            print(f"Error applying LIME for report: {e}")
            plt.subplot(2, 2, 3)
            plt.text(0.5, 0.5, "LIME explanation\nnot available", 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.axis('off')
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig()
        plt.close()
        
        # Explanations page 2 - Integrated Gradients and CAM
        plt.figure(figsize=(8.5, 11))
        plt.suptitle("Explanation Methods - Part 2", fontsize=16, y=0.98)
        
        # Try Integrated Gradients
        try:
            ig_attrs = apply_integrated_gradients(
                model, image_tensor, original_image,
                target_class=pred_class, device=device
            )
            
            # Create a visualization using the attribution map
            cmap = plt.get_cmap(COLORMAP)
            ig_heatmap = cmap(ig_attrs)[:, :, :3]
            
            # Resize to match original image
            ig_heatmap_resized = cv2.resize(
                ig_heatmap, (original_image.shape[1], original_image.shape[0])
            )
            
            # Create overlay
            ig_overlay = (1 - OVERLAY_ALPHA) * original_image + OVERLAY_ALPHA * ig_heatmap_resized
            ig_overlay = np.clip(ig_overlay, 0, 1)
            
            plt.subplot(2, 2, 1)
            plt.imshow(ig_overlay)
            plt.title("Integrated Gradients Explanation")
            plt.axis('off')
            
            plt.subplot(2, 2, 2)
            plt.imshow(ig_attrs, cmap=COLORMAP)
            plt.title("IG Attribution Map")
            plt.axis('off')
            
            ig_description = (
                "Integrated Gradients computes the path integral of the gradients "
                "along a straight line from a baseline image (usually black) to "
                "the input image. This provides a pixel-level attribution map "
                "showing the contribution of each pixel to the prediction."
            )
            
            plt.figtext(0.1, 0.55, "Integrated Gradients Explanation:", fontsize=12, weight='bold')
            plt.figtext(0.1, 0.5, ig_description, fontsize=10, wrap=True)
            
        except Exception as e:
            print(f"Error applying Integrated Gradients for report: {e}")
            plt.subplot(2, 2, 1)
            plt.text(0.5, 0.5, "Integrated Gradients\nexplanation not available", 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.axis('off')
            
            plt.subplot(2, 2, 2)
            plt.axis('off')
        
        # Try CAM
        try:
            cam_heatmap, cam_overlay = apply_cam(
                model, image_tensor, original_image,
                target_class=pred_class, device=device
            )
            
            plt.subplot(2, 2, 3)
            plt.imshow(cam_overlay)
            plt.title("Class Activation Mapping")
            plt.axis('off')
            
            plt.subplot(2, 2, 4)
            plt.imshow(cam_heatmap, cmap=COLORMAP)
            plt.title("CAM Heatmap")
            plt.axis('off')
            
            cam_description = (
                "Class Activation Mapping (CAM) visualizes the class-specific "
                "feature maps in the last convolutional layer. It shows which "
                "regions of the image were most important for the model's "
                "prediction of the specific class."
            )
            
            plt.figtext(0.1, 0.25, "Class Activation Mapping Explanation:", fontsize=12, weight='bold')
            plt.figtext(0.1, 0.2, cam_description, fontsize=10, wrap=True)
            
        except Exception as e:
            print(f"Error applying CAM for report: {e}")
            plt.subplot(2, 2, 3)
            plt.text(0.5, 0.5, "CAM explanation\nnot available", 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.axis('off')
            
            plt.subplot(2, 2, 4)
            plt.axis('off')
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig()
        plt.close()
        
        # Final page with disclaimer
        plt.figure(figsize=(8.5, 11))
        plt.suptitle("Important Information", fontsize=16, y=0.98)
        
        disclaimer_text = (
            "DISCLAIMER:\n\n"
            "This report was generated by an artificial intelligence system for research purposes only. "
            "The predictions and explanations provided should not be used for clinical diagnosis or "
            "treatment decisions without proper medical supervision.\n\n"
            "The model was trained on the HAM10000 dataset, which may not be representative of all "
            "skin types, conditions, or demographics. Performance can vary significantly based on "
            "image quality, lighting conditions, and skin characteristics.\n\n"
            "Explanation methods like Grad-CAM, LIME, and Integrated Gradients provide insight into "
            "the model's decision-making process, but they are approximations and should be interpreted "
            "with caution.\n\n"
            "For any concerns about skin lesions, please consult a qualified dermatologist. Early "
            "detection and proper medical assessment are crucial for skin cancer diagnosis and treatment."
        )
        
        plt.figtext(0.1, 0.9, disclaimer_text, fontsize=12, wrap=True)
        
        about_model_text = (
            "About the Model:\n\n"
            f"Architecture: {model.__class__.__name__}\n"
            "Training Dataset: HAM10000\n"
            "Classes: Melanocytic nevi (nv), Melanoma (mel), Benign keratosis (bkl), "
            "Basal cell carcinoma (bcc), Actinic keratoses (akiec), Vascular lesions (vasc), "
            "Dermatofibroma (df)\n\n"
            "Explanations generated using multiple XAI (Explainable AI) techniques "
            "to provide a comprehensive understanding of the model's prediction."
        )
        
        plt.figtext(0.1, 0.4, about_model_text, fontsize=12, wrap=True)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig()
        plt.close()
    
    # Return paths and information
    report_data = {
        'summary_path': summary_path,
        'comprehensive_path': comprehensive_path,
        'prediction': {
            'class_index': pred_class,
            'class_name': pred_name,
            'confidence': pred_prob
        },
        'top_predictions': [
            {'class_index': cls, 'class_name': class_names[cls], 'confidence': prob}
            for cls, prob in zip(top_classes, top_probs)
        ]
    }
    
    return report_data