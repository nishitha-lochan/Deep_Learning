# 🧠 CIFAR-10 Image Classification using CNN

This project demonstrates image classification on the **CIFAR-10 dataset** using both **Artificial Neural Networks (ANN)** and **Convolutional Neural Networks (CNN)**. The results clearly show how CNN significantly outperforms traditional ANN for image-based tasks.

---

## 📌 Project Overview

The goal of this project is to classify 32×32 color images into **10 different object categories** using deep learning techniques.

### CIFAR-10 Classes:

* Airplane
* Automobile
* Bird
* Cat
* Deer
* Dog
* Frog
* Horse
* Ship
* Truck

---

## 🚀 Features Implemented

* Dataset loading and preprocessing
* Image normalization
* ANN baseline model
* CNN architecture implementation
* Model training and evaluation
* Accuracy and loss visualization
* Confusion matrix visualization

---

## 🛠 Technologies Used

* Python
* TensorFlow / Keras
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn

---

## 📂 Dataset Information

**Dataset:** CIFAR-10

* Training Images: 50,000
* Testing Images: 10,000
* Image Size: 32 × 32 × 3 (RGB)
* Classes: 10

Dataset is directly loaded using TensorFlow Keras API.

---

## 🏗 Model Architectures

### Artificial Neural Network (ANN)

```
Input → Flatten → Dense → Dense → Output
```

### Convolutional Neural Network (CNN)

```
Input Image
     ↓
Conv2D + ReLU
     ↓
MaxPooling
     ↓
Conv2D + ReLU
     ↓
MaxPooling
     ↓
Flatten
     ↓
Dense
     ↓
Softmax Output
```

---

## 📊 Results

| Model | Test Accuracy |
| ----- | ------------- |
| ANN   | ~49.9%        |
| CNN   | ⭐ ~81.2%      |

CNN provides a significant performance improvement by preserving spatial features and learning hierarchical patterns.

---
## 🎯 Learning Outcomes

Through this project, I learned:

* CNN architecture design
* Feature extraction using convolution layers
* Pooling and dimensionality reduction
* Image preprocessing techniques
* Performance comparison between ANN and CNN
* Model evaluation and visualization

---

## 🔮 Future Improvements

* Add data augmentation
* Implement dropout and batch normalization
* Try deeper CNN architectures
* Apply transfer learning (ResNet, VGG)
* Deploy model using Streamlit

---


