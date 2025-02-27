# 🧠 Explainable AI (XAI) for Image Classification

This repository provides implementations of four **Explainable AI (XAI) methods** for interpreting deep learning models in image classification. These methods generate **heatmaps** that highlight the most important regions of an image contributing to the model's prediction.

## 🚀 Methods Implemented
1. **Class Activation Mapping (CAM)**
2. **Grad-CAM (Gradient-weighted Class Activation Mapping)**
3. **Grad-CAM++**
4. **Score-CAM (Score-weighted CAM)**

Each of these methods helps visualize how a deep learning model (e.g., **ResNet50**) makes decisions, improving interpretability for AI practitioners, researchers, and domain experts.


## 📌 1. Class Activation Mapping (CAM)
**📖 Description:**  

- **CAM** is an early explainability method that highlights important image regions based on **global average pooling (GAP)** applied to convolutional feature maps.

- It requires **modifying the CNN architecture** to replace fully connected (FC) layers with a **GAP layer**, making it applicable **only to specific network architectures**.

**📊 How It Works:**

1. Uses **feature maps** from the last convolutional layer.
2. Applies **GAP** to obtain class-specific weights.
3. Computes a weighted sum of the activation maps.

**⚠️ Limitations:**

- Requires architectural modification, making it incompatible with pre-trained models.
- Less accurate than later methods like **Grad-CAM**.

---

## 📌 2. Grad-CAM (Gradient-weighted CAM)
**📖 Description:**  

- **Grad-CAM** improves CAM by using **gradients of the target class** to **weigh the activation maps**, highlighting key regions **without modifying the network architecture**.
- It is **compatible with any CNN-based model**.

**📊 How It Works:**

1. Captures **feature maps** from the last convolutional layer.
2. Computes **gradients** of the target class score **w.r.t.** these feature maps.
3. Performs a **weighted sum** of the feature maps using the computed gradients.
4. Applies **ReLU activation** to focus only on important regions.

**✅ Advantages:**

- **Works with any CNN model** without modifications.
- Provides **class-specific heatmaps**, useful for interpreting model decisions.

**⚠️ Limitations:**

- Can miss **fine-grained** details in complex images.
- Focuses only on **positive influences**, ignoring negative contributions.

---

## 📌 3. Grad-CAM++ (Enhanced Grad-CAM)
**📖 Description:**  

- **Grad-CAM++** is an improvement over Grad-CAM that assigns **better weight distributions** to activation maps, improving localization.
- It captures **multiple important regions**, making it more precise for **overlapping objects**.

**📊 How It Works:**

1. Computes **first-order and second-order gradients**.
2. Uses these gradients to **refine the weighting** of activation maps.
3. Generates **more localized and precise heatmaps**.

**✅ Advantages:**

- **Better localization** than Grad-CAM.
- Captures **multiple regions** of interest instead of just one.
- More robust for **complex images**.

**⚠️ Limitations:**

- **Higher computational cost** than Grad-CAM.
- Requires computing **higher-order gradients**.

---

## 📌 4. Score-CAM (Score-weighted CAM)
**📖 Description:**  

- **Score-CAM** removes the need for **gradients**, making it a **gradient-free** interpretability method.
- It **perturbs the input image** using feature maps and measures the change in the model's confidence score.

**📊 How It Works:**

1. Extracts **activation maps** from the last convolutional layer.
2. **Perturbs** the original image by multiplying it with each activation map.
3. Computes the **model's confidence score** for each perturbed image.
4. Uses these confidence scores to **weight the activation maps**.

**✅ Advantages:**

- **Does not require gradients**, making it model-agnostic.
- Produces **better heatmap localization** than Grad-CAM.
- Works well with **any CNN model**.

**⚠️ Limitations:**

- **Computationally expensive**, as it requires multiple forward passes.
- Sensitive to **perturbation variations**.

---

## 📌 Comparison of XAI Methods

| Method     | Requires Gradients? | Model Modification? | Computational Cost | Localization Quality |
|------------|-------------------|-------------------|------------------|---------------------|
| **CAM**       | ❌ No  | ✅ Yes  | ⭐ Fast  | ⚠️ Low |
| **Grad-CAM**  | ✅ Yes | ❌ No   | ⭐⭐ Medium | ⭐⭐ Good |
| **Grad-CAM++**| ✅ Yes | ❌ No   | ⭐⭐⭐ High | ⭐⭐⭐ Better |
| **Score-CAM** | ❌ No  | ❌ No   | ⭐⭐⭐⭐ Very High | ⭐⭐⭐⭐ Best |

---

## 🛠 How to Run

### **🔹 Setup**
1. Install dependencies:
   ```bash
   pip install torch torchvision numpy opencv-python matplotlib
   ```

2. Clone the repository:
   ```bash
   git clone https://github.com/PawanRamaMali/XAI.git
   cd XAI
   ```

3. Run any of the methods:
   ```bash
   python grad_cam.py  # For Grad-CAM
   python grad_cam_plus.py  # For Grad-CAM++
   python score_cam.py  # For Score-CAM
   ```
---


## 🎯 **Use Cases**
✅ **Medical Imaging:** Identifying critical regions in **X-rays, MRIs, and CT scans**.  
✅ **Autonomous Vehicles:** Understanding how a model detects **traffic signs, pedestrians, and obstacles**.  
✅ **Security & Forensics:** Interpreting **face recognition models and fraud detection systems**.  
✅ **AI Fairness & Bias Detection:** Ensuring models focus on **relevant, unbiased features**.  

---


### **🔹 Summary of Explainability Methods in XAI**  

This table provides a high-level **comparison of different XAI methods**, including their **type, key approach, best use case, and pros & cons.**  

| **Method**           | **Type**       | **Approach** | **Best For** | **Pros ✅** | **Cons ❌** |
|----------------------|---------------|-------------|-------------|------------|------------|
| **CAM (Class Activation Mapping)** | Local | Feature map weighting | **CNNs** | Fast, simple | Requires network modification |
| **Grad-CAM** | Local | Gradients | **CNNs, Image Classification** | Works on pre-trained models | Can be coarse |
| **Grad-CAM++** | Local | Improved Gradients | **CNNs, Multiple Objects** | Better localization | More computationally expensive |
| **Score-CAM** | Local | Perturbation-based | **CNNs (No gradients needed)** | Model-agnostic | Computationally expensive |
| **LIME** | Local | Perturbation & Surrogate Model | **Any Model, NLP, Tabular, Images** | Model-agnostic | Can be unstable, only local |
| **SHAP (Shapley Values)** | Local & Global | Game Theory | **Any Model, Global Interpretability** | Fair, consistent feature attribution | Computationally expensive |
| **Kernel SHAP** | Local & Global | Approximate SHAP | **Large Datasets, ML Models** | Faster than SHAP | Less accurate |
| **Integrated Gradients** | Local | Gradient-based | **Deep Learning, NLP** | Captures non-linearity | Requires differentiability |
| **DeepLIFT** | Local | Reference-based Activation Difference | **Deep Learning, Medical AI** | Faster than IG | Requires reference selection |
| **Permutation Importance** | Global | Feature Shuffling | **Feature Selection, ML Models** | Simple, fast | Doesn't capture feature interactions |
| **Occlusion Sensitivity** | Local | Masking & Perturbation | **CNNs, Medical Imaging** | Works on any CNN | Computationally expensive |
| **Contrastive Explanations Method (CEM)** | Local | Contrastive Learning | **Bias Detection, NLP** | Finds minimal feature changes | Computationally expensive |
| **XGBoost Feature Importance** | Global | Gain-based ranking | **Tree-Based Models** | Built-in, fast | Doesn’t explain interactions |

---

## **🔹 Which Method Should You Use?**
### **1️⃣ Explaining Deep Learning Models (CNNs & NLP)**
| Use Case | Best Methods |
|----------|-------------|
| **Explaining CNN decisions (Image AI)** | **Grad-CAM, Score-CAM, Occlusion Sensitivity** |
| **Explaining multiple objects in an image** | **Grad-CAM++, Score-CAM** |
| **Explaining NLP models** | **Integrated Gradients, DeepLIFT, SHAP** |

### **2️⃣ Explaining Any Machine Learning Model**
| Use Case | Best Methods |
|----------|-------------|
| **Feature Importance for Any Model** | **SHAP, LIME, Permutation Importance** |
| **Ensuring fairness & bias detection** | **SHAP, CEM, Permutation Importance** |
| **Explaining ensemble models (XGBoost, Random Forests)** | **SHAP, XGBoost Feature Importance** |

### **3️⃣ Fast Interpretability & Feature Selection**
| Use Case | Best Methods |
|----------|-------------|
| **Quick interpretation of black-box models** | **LIME, SHAP, Kernel SHAP** |
| **Feature selection for better performance** | **Permutation Importance, SHAP** |

---

## **🔹 Key Takeaways**
1. **Grad-CAM & Score-CAM** → Best for **CNNs (image models)**.  
2. **SHAP** → Best for **global & local explanations (fair and consistent)**.  
3. **LIME** → Good for **fast, local interpretations** but less stable.  
4. **Integrated Gradients & DeepLIFT** → Best for **deep learning models (NLP, medical AI)**.  
5. **Permutation Importance** → Simple but effective for **feature selection**.  

---

## 📜 License
This project is licensed under the **MIT License**.

---

## 📬 Contact
For questions or collaborations, feel free to reach out:
- 📧 Email: prm@outlook.in

