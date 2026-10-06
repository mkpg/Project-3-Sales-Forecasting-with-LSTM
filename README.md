                    ┌─────────────────────────┐
                    │   Clean Sales Dataset   │
                    │     cleaned_sales.csv   │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │    Data Preparation     │
                    │                         │
                    │ • Group by Order_Date   │
                    │ • Calculate daily sales │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │   Time-Series Windows   │
                    │                         │
                    │ Previous 7 days ──►     │
                    │ Predict next day        │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │   Chronological Split   │
                    │                         │
                    │ 70% Training            │
                    │ 30% Testing             │
                    │                         │
                    │ No random shuffling     │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │      Data Scaling       │
                    │                         │
                    │      MinMaxScaler       │
                    │                         │
                    │ Train → fit_transform   │
                    │ Test  → transform       │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │    Input Reshaping      │
                    │                         │
                    │ (samples, 7, 1)         │
                    │                         │
                    │ 7 time steps            │
                    │ 1 feature: Net_Sales    │
                    └────────────┬────────────┘
                                 │
                                 ▼
             ┌─────────────────────────────────────┐
             │             LSTM MODEL               │
             │                                     │
             │          LSTM(50 units)             │
             │                ↓                    │
             │           Dense(1)                  │
             │                                     │
             │     Optimizer: Adam                 │
             │     Loss: MSE                       │
             │     Metrics: MAE, RMSE              │
             └──────────────────┬──────────────────┘
                                │
                                ▼
                    ┌─────────────────────────┐
                    │       Prediction        │
                    │                         │
                    │ Predict scaled sales    │
                    │          ↓              │
                    │ Inverse transform       │
                    │          ↓              │
                    │ Actual ₹ sales values   │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │      Evaluation         │
                    │                         │
                    │ • MAE                   │
                    │ • RMSE                  │
                    │ • Actual vs Predicted   │
                    │   visualization         │
                    └─────────────────────────┘

1. Data Preparation
Converts transaction-level sales data into a daily time series because the LSTM requires sequential observations.

2. Sliding Window
Uses the previous 7 days of sales as the input sequence and the following day as the prediction target.

3. Chronological Split
The dataset is split according to time rather than randomly to prevent future information from leaking into the training data.

4. Scaling
MinMaxScaler normalizes the sales values so the neural network can optimize more effectively. The scaler is fitted only on training data.

5. LSTM
The LSTM learns temporal patterns from the sequence of previous sales values and produces a single next-day sales prediction.

6. Inverse Scaling
Model outputs are converted back from normalized values into the original sales scale for meaningful interpretation.

7. Evaluation
MAE and RMSE measure prediction error, while the actual-vs-predicted plot provides a visual comparison of model performance.
