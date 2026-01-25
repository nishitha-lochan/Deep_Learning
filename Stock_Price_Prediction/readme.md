# **Stock Price Prediction Using RNN and LSTM**

## **Project Overview**

This project demonstrates **time series forecasting** using **Recurrent Neural Networks (RNN)** and **Long Short-Term Memory (LSTM)** networks. The goal is to **predict future stock prices** based on historical data.

The project also compares the **performance of RNN vs LSTM** on a small dataset to highlight how model choice depends on **dataset size and sequence length**.

---

## **Dataset**

* Dataset contains daily prices of 5 stocks for 1 year (365 days).
* Sample columns: `Stock_1, Stock_2, Stock_3, Stock_4, Stock_5`.
* Only numeric stock prices are used for prediction; the date column is dropped.
* **Example rows**:

| Stock_1 | Stock_2 | Stock_3 | Stock_4 | Stock_5 |
| ------- | ------- | ------- | ------- | ------- |
| 101.76  | 100.16  | 99.49   | 99.90   | 101.76  |
| 102.17  | 99.97   | 98.68   | 100.64  | 102.52  |

---

## **Methodology**

1. **Data Preprocessing**

   * Drop non-numeric columns (Date)
   * Scale stock prices using `MinMaxScaler` (0–1 range)
   * Create sequences of **60 past days** to predict the next day

2. **Modeling**

   * **Simple RNN** with 50 units, `tanh` activation, and Dense output layer
   * **LSTM** with 50 units, `tanh` activation, and Dense output layer

3. **Training**

   * 80% train, 20% test split
   * 25 epochs, batch size = 16
   * Loss: Mean Squared Error

4. **Evaluation**

   * Metrics: **RMSE** (Root Mean Squared Error), **MAE** (Mean Absolute Error)
   * Visualizations: Actual vs Predicted prices, Prediction Error per day

---

## **Results (TIME_STEPS = 60)**

| Model | RMSE   | MAE    |
| ----- | ------ | ------ |
| RNN   | 1.3401 | 1.0244 |
| LSTM  | 1.8757 | 1.5206 |

**Insights:**

* RNN slightly outperformed LSTM on this small dataset.
* LSTM generally performs better with **longer sequences or larger datasets**, but here RNN captures short-term trends effectively.

---

## **Visualizations**

* **Actual vs Predicted Prices** – compare RNN and LSTM predictions with real stock prices
* **Prediction Errors** – shows under/over predictions for each model


---

## **Future Improvements**

* Use **all 5 stocks as features** (multivariate input)
* Test different **sequence lengths** and **hidden units**
* Add **dropout** or **regularization** to prevent overfitting
* Increase dataset size (more historical data)

---

## **Technologies Used**

* Python, NumPy, Pandas, Matplotlib
* Scikit-learn (for scaling and metrics)
* TensorFlow / Keras (RNN and LSTM models)

---
