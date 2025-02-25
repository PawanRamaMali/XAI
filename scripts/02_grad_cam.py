import torch
import cv2
import numpy as np
from torchvision import models, transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image

# Load pre-trained ResNet50 model
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
model.eval()

# Store activations and gradients
activations = None
gradients = None

# Hook function to capture forward activations
def forward_hook(module, input, output):
    global activations
    activations = output

# Hook function to capture gradients
def backward_hook(module, grad_in, grad_out):
    global gradients
    gradients = grad_out[0]  # Store the first element (gradients w.r.t. output)

# Register hooks on the last convolutional layer
target_layer = model.layer4[2].conv3  # Last conv layer in ResNet50
target_layer.register_forward_hook(forward_hook)
target_layer.register_full_backward_hook(backward_hook)

# Preprocess image
def preprocess_image(img_path):
    img = Image.open(img_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(img).unsqueeze(0)

# Compute Grad-CAM
def compute_gradcam(img_path, class_idx):
    img = preprocess_image(img_path)
    output = model(img)
    model.zero_grad()
    output[0, class_idx].backward()

    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    for i in range(activations.shape[1]):
        activations[:, i, :, :] *= pooled_gradients[i]
    
    heatmap = torch.mean(activations, dim=1).squeeze().detach().numpy()
    heatmap = np.maximum(heatmap, 0)  # ReLU operation
    heatmap /= np.max(heatmap)  # Normalize
    return heatmap

# Overlay heatmap on original image
def overlay_heatmap(img_path, heatmap):
    img = cv2.imread(img_path)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    return overlay

if __name__ == "__main__":
    img_path = "car002.jpg"  # Ensure this file exists
    class_idx = 243  # Example class (Boxer dog in ImageNet)

    heatmap = compute_gradcam(img_path, class_idx)
    overlay = overlay_heatmap(img_path, heatmap)

    cv2.imshow("Grad-CAM", overlay)
    cv2.imwrite("grad_cam_output.jpg", overlay)  # Save the output image
    cv2.waitKey(0)
    cv2.destroyAllWindows()
