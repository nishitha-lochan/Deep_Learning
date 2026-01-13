# Handwritten Digit Recognition using Deep Learning (MNIST)

## 📌 Project Overview

This project focuses on building a **Handwritten Digit Recognition system** using the **MNIST dataset** and **Deep Learning** techniques. The goal is to classify grayscale images of handwritten digits (0–9) by training a neural network with hidden layers and evaluating its performance using standard classification metrics.

The model is implemented using **Deep Learning (Neural Networks)** with **ReLU** and **Sigmoid** activation functions, and its performance is analyzed using **confusion matrix and accuracy metrics**.

---

## 🧠 Dataset

* **Dataset Name:** MNIST Handwritten Digits
* **Images:** 28×28 grayscale images
* **Classes:** 10 (Digits 0–9)
* **Training Samples:** 60,000
* **Test Samples:** 10,000

---

## ⚙️ Technologies Used

* Python
* TensorFlow / Keras
* NumPy
* Matplotlib
* Scikit-learn

---

## 🏗️ Model Architecture

* Input Layer: 784 neurons (28×28 flattened image)
* Hidden Layers:

  * Dense Layer with **ReLU activation**
  * (Optional additional hidden layers for better feature learning)
* Output Layer:

  * 10 neurons with **Sigmoid activation**

---

## 🔄 Activation Functions

* **ReLU (Rectified Linear Unit):**

  * Used in hidden layers for faster convergence and efficient feature extraction.

* **Sigmoid:**

  * Used in the output layer to generate probability-like outputs for classification.

---

## 🏋️ Model Training

* Loss Function: Categorical Cross-Entropy
* Optimizer: Adam
* Epochs: Configurable
* Batch Size: Configurable

---

## 📊 Model Evaluation

The model performance is evaluated using:

* Accuracy score
* Confusion Matrix

### Confusion Matrix

The confusion matrix helps analyze:

* Correct predictions for each digit
* Misclassification patterns between similar digits (e.g., 4 and 9, 3 and 5)

---

## 🔍 Results

* Achieved high classification accuracy on the MNIST test dataset
* Demonstrated effective learning using hidden layers
* ReLU activation improved training speed
* Confusion matrix provided clear insight into prediction performance

---

## 🚀 How to Run the Project

1. Clone the repository

   ```bash
   git clone https://github.com/your-username/mnist-digit-classification.git
   ```
2. Install required libraries

   ```bash
   pip install -r requirements.txt
   ```
3. Run the training script

   ```bash
   python train.py
   ```
4. Evaluate the model

   ```bash
   python evaluate.py
   ```

---

## 📈 Future Improvements

* Replace Sigmoid with Softmax for better multi-class probability distribution
* Add CNN layers for improved accuracy
* Hyperparameter tuning
* Deploy as a web application

---

