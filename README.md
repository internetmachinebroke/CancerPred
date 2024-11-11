# 🧠 Glioblastoma Detection Using Deep Learning

A deep learning model for detecting glioblastoma brain tumors from MRI scans using TensorFlow and Keras. This project implements a Convolutional Neural Network (CNN) to analyze brain MRI images and identify the presence of glioblastoma tumors with high accuracy.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-orange.svg)](https://tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

## 📋 Table of Contents
- [Overview](#-overview)
- [Dataset](#-dataset)
- [Model Architecture](#-model-architecture)
- [Performance](#-performance)
- [Requirements](#-requirements)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Model Details](#-model-details)
- [Results](#-results)
- [License](#-license)

## 🔍 Overview
This project implements a deep learning solution for detecting glioblastoma in brain MRI scans. The model is designed to assist medical professionals in early tumor detection and classification.

### Key Features
- Advanced CNN architecture optimized for medical imaging
- Data augmentation to improve model robustness
- Confidence scoring for predictions
- High accuracy and AUC metrics
- Support for grayscale MRI images (256x256x1)

## 📊 Dataset
The project uses the MRI Glioma Dataset for Tumor Grade Classification by Kolar Luni, available on Kaggle. The dataset includes:
- 9,832 MRI brain tumor images
- Image dimensions: 256x256x1 (grayscale)
- All images are pre-processed and standardized

## 🏗️ Model Architecture
The CNN architecture consists of:
```
- Input Layer (256x256x1)
- 3 Convolutional Blocks:
  └─ Each block contains:
     - 2x Conv2D layers with ReLU
     - BatchNormalization
     - MaxPooling
     - Dropout (0.25)
- Dense Layers:
  - Flatten
  - Dense (512) + BatchNorm + Dropout
  - Dense (256) + BatchNorm + Dropout
  - Output Dense (1) with Sigmoid
```

## 📈 Performance
- Accuracy: 95.73%
- AUC: 97.87%
- Loss: 0.1116

## 💻 Requirements
```
tensorflow>=2.0.0
pillow
numpy
pandas
scikit-learn
tqdm
matplotlib
seaborn
```

## 🚀 Installation
1. Clone the repository
```bash
git clone https://github.com/internetmachinebroke/CancerPred.git
cd CancerPred
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Prepare your data
```bash
# Place your MRI images in the GT_path directory
mkdir GT_path
# Copy your images to GT_path/
```

## 🔧 Usage
To train the model:
```bash
python main.py
```

To make predictions on new images:
```python
from predictor import predict_single_image
from tensorflow.keras.models import load_model

model = load_model('best_model.keras')
result = predict_single_image(model, 'path_to_image.jpg')
print(f"Tumor detected: {'Yes' if result['has_tumor'] else 'No'}")
print(f"Confidence: {result['confidence']:.2f}%")
```

## 📁 Project Structure
```
CancerPred/
├── GT_path/              # Directory for MRI images
├── main.py              # Main execution script
├── data_loader.py       # Data loading and augmentation
├── data_processor.py    # Data preprocessing
├── model.py            # CNN model architecture
├── trainer.py          # Model training logic
├── visualizer.py       # Training visualization
├── predictor.py        # Image prediction
└── requirements.txt    # Project dependencies
```

## 🔬 Model Details
The model employs several key techniques:
- Data augmentation for improved generalization
- Batch normalization for training stability
- Dropout layers to prevent overfitting
- Early stopping and learning rate reduction
- Confidence scoring for predictions

### Training Parameters
- Batch Size: 32
- Initial Learning Rate: 0.001
- Optimizer: Adam
- Loss Function: Binary Cross-entropy
- Metrics: Accuracy, AUC

## 📊 Results
The model achieves:
- High accuracy in tumor detection (95.73%)
- Excellent discrimination ability (AUC: 97.87%)
- Low loss value (0.1116)
- Robust performance on validation data

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments
- Dataset provided by Kolar Luni on Kaggle

---
*Note: This model is intended for research purposes only and should not be used as the sole means of medical diagnosis.*
