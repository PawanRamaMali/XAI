import torch
import cv2
import numpy as np
from torchvision import models, transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image

# Load pre-trained ResNet50 model
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
model.eval()

# Store activations
activations = None

# Hook function to capture forward activations
def forward_hook(module, input, output):
    global activations
    activations = output

# Register hook on the last convolutional layer
target_layer = model.layer4[2].conv3  # Last conv layer in ResNet50
target_layer.register_forward_hook(forward_hook)

# Preprocess image
def preprocess_image(img_path):
    img = Image.open(img_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(img).unsqueeze(0)

# Compute Score-CAM
def compute_score_cam(img_path, class_idx):
    img = preprocess_image(img_path)
    _ = model(img)
    
    activation_maps = activations.squeeze(0).detach().numpy()
    num_channels = activation_maps.shape[0]
    scores = []
    
    for i in range(num_channels):
        upsampled_map = cv2.resize(activation_maps[i], (224, 224))
        upsampled_map = (upsampled_map - np.min(upsampled_map)) / (np.max(upsampled_map) - np.min(upsampled_map))
        
        weighted_img = img.clone()
        weighted_img[:, 0, :, :] *= torch.tensor(upsampled_map)
        weighted_img[:, 1, :, :] *= torch.tensor(upsampled_map)
        weighted_img[:, 2, :, :] *= torch.tensor(upsampled_map)
        
        output = model(weighted_img)
        scores.append(output[0, class_idx].item())
    
    scores = np.array(scores)
    scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
    
    score_cam_map = np.sum(activation_maps * scores[:, np.newaxis, np.newaxis], axis=0)
    score_cam_map = np.maximum(score_cam_map, 0)  # ReLU operation
    score_cam_map /= np.max(score_cam_map)  # Normalize
    return score_cam_map

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

    heatmap = compute_score_cam(img_path, class_idx)
    overlay = overlay_heatmap(img_path, heatmap)

    cv2.imshow("Score-CAM", overlay)
    cv2.imwrite("score_cam_output.jpg", overlay)  # Save the output image
    cv2.waitKey(0)
    cv2.destroyAllWindows()
