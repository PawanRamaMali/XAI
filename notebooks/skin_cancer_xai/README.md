## Project Structure 

```

skin_cancer_xai/
├── README.md               # Project overview and setup instructions
├── requirements.txt        # Required packages for installation
├── config.py               # Configuration parameters
├── main.py                 # Main execution script
├── data/
│   ├── __init__.py
│   ├── dataset.py          # Dataset class definitions
│   └── preprocessing.py    # Data preprocessing functions
├── models/
│   ├── __init__.py
│   └── cnn_model.py        # Model architecture definitions
├── training/
│   ├── __init__.py
│   ├── train.py            # Training functions
│   └── evaluate.py         # Evaluation functions
├── explanation/
│   ├── __init__.py
│   ├── gradcam.py          # Grad-CAM implementation
│   ├── lime_explainer.py   # LIME implementation
│   ├── integrated_gradients.py  # Integrated Gradients implementation 
│   ├── shap_explainer.py   # SHAP implementation
│   └── cam.py              # Class Activation Mapping implementation
└── visualization/
    ├── __init__.py
    ├── plotting.py         # General plotting functions
    └── clinical_report.py  # Clinical decision support visualizations

```

# Explainable AI for Skin Cancer Detection

This project demonstrates various explainability techniques for deep learning models used in skin cancer detection. It provides a comprehensive implementation of multiple XAI methods using PyTorch.

## Features

- Loading and preprocessing dermatology image datasets
- CNN model implementation with transfer learning (ResNet, EfficientNet)
- Model training and evaluation pipeline
- Implementation of multiple explainability methods:
  - Grad-CAM
  - LIME
  - Integrated Gradients
  - SHAP
  - Class Activation Mapping
- Visualization tools for clinical interpretation

## Dataset

This implementation is designed to work with the HAM10000 ("Human Against Machine with 10000 training images") dataset, which contains dermatoscopic images of pigmented skin lesions across seven diagnostic categories:
- Actinic keratoses (akiec)
- Basal cell carcinoma (bcc)
- Benign keratosis (bkl)
- Dermatofibroma (df)
- Melanoma (mel)
- Melanocytic nevi (nv)
- Vascular lesions (vasc)

The dataset can be downloaded from:
- [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T)
- [Kaggle](https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000)

After downloading, place the images in a directory structure as follows:
```
data/
├── HAM10000_metadata.csv
└── HAM10000_images/
    ├── ISIC_0024306.jpg
    ├── ISIC_0024307.jpg
    └── ...
```

## Prerequisites

1. Python 3.8 or higher
2. Dependencies listed in `requirements.txt`

## Installation

1. Clone this repository:
```bash
git clone https://github.com/username/skin-cancer-xai.git
cd skin-cancer-xai
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Update the configuration in `config.py` to match your dataset paths.

2. Run the main script:
```bash
python main.py
```

## Components

- `data/`: Dataset handling and preprocessing
- `models/`: Model architecture definitions
- `training/`: Training and evaluation functions
- `explanation/`: Implementations of XAI methods
- `visualization/`: Plotting and visualization tools

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- The HAM10000 dataset: Tschandl, P., Rosendahl, C. & Kittler, H. The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. Sci. Data 5, 180161 doi:10.1038/sdata.2018.161 (2018).