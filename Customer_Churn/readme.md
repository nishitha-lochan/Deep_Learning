# 📊 Customer Churn Prediction Using ANN (TensorFlow & Keras)

This project builds an **Artificial Neural Network (ANN)** model to predict customer churn for a telecom company. The model identifies customers who are likely to leave the service, helping businesses take proactive retention actions.

The project handles **class imbalance**, applies **threshold tuning**, and evaluates performance using industry-standard metrics such as **ROC-AUC, Precision, Recall, and F1-score**.

---

##  Project Highlights

* Built using **TensorFlow & Keras**
* Handled **imbalanced dataset**
* Optimized classification threshold
* Achieved strong churn detection performance
* Business-oriented evaluation metrics

---

## 📁 Dataset

**Telco Customer Churn Dataset**

Key features include:

* Customer demographics
* Subscription services
* Contract details
* Billing information
* Internet & phone services

Target column:

* `Churn` (Yes / No)

---

## 🧠 Model Architecture

ANN structure:

* Input Layer → Feature Inputs
* Hidden Layer 1 → 64 Neurons (ReLU)
* Hidden Layer 2 → 32 Neurons (ReLU)
* Output Layer → 1 Neuron (Sigmoid)

Additional Techniques:

* Class Weighting
* Threshold Optimization
* Feature Scaling
* One-hot Encoding

---

## ⚙️ Technologies Used

* Python
* TensorFlow
* Keras
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Joblib

---

## 📈 Model Performance

### Classification Results:

```
Accuracy: 71%

Churn Class (1):
Precision: 48%
Recall: 86%
F1-score: 61%

ROC-AUC Score: 0.84
```

---

## 🎯 Why These Metrics Matter

* **High Recall (86%)**
  → Successfully identifies most churn customers

* **ROC-AUC (0.84)**
  → Strong ability to distinguish churn vs non-churn customers

* **Balanced Performance**
  → Optimized for business use cases rather than misleading accuracy

---

## 🔍 Threshold Optimization

Instead of default `0.5`, the classification threshold was tuned:

```python
y_pred = (y_prob > 0.45)
```

This improves churn detection while maintaining stable overall accuracy.


## 📊 Business Use Case

This model helps telecom companies:

* Identify customers at risk of churn
* Launch targeted retention campaigns
* Reduce revenue loss
* Improve customer satisfaction

---

## 📌 Future Improvements

* Hyperparameter tuning
* Feature selection
* XGBoost model comparison
* Deployment using Flask / FastAPI
* Dashboard visualization

