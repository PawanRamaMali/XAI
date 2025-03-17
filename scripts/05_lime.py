# Install LIME if not installed
# pip install lime

import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models, transforms
from PIL import Image
from lime import lime_image
from skimage.segmentation import mark_boundaries

# Load pre-trained ResNet50 model
model = models.resnet50(pretrained=True)
model.eval()

# Preprocessing function
def preprocess_image(img_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = Image.open(img_path).convert('RGB')
    return transform(img).unsqueeze(0)

# Function to predict class probabilities
def predict(img_array):
    img_tensor = torch.tensor(img_array.transpose(0, 3, 1, 2)).float()
    img_tensor = img_tensor / 255.0  # Normalize pixel values
    img_tensor = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img_tensor)
    with torch.no_grad():
        output = model(img_tensor)
        return torch.nn.functional.softmax(output, dim=1).numpy()

# Load image
img_path = "car002.jpg"  # Change with your image
img = Image.open(img_path).convert('RGB')

# Create LIME explainer
explainer = lime_image.LimeImageExplainer()

# Explain prediction
explanation = explainer.explain_instance(
    np.array(img), predict, top_labels=1, hide_color=0, num_samples=1000
)

# Get mask for the top class
top_label = explanation.top_labels[0]
temp, mask = explanation.get_image_and_mask(
    top_label, positive_only=True, num_features=5, hide_rest=True
)

# Display LIME explanation
plt.figure(figsize=(8, 8))
plt.imshow(mark_boundaries(temp, mask))
plt.title("LIME Explanation")
plt.axis("off")
plt.show()
