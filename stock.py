import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

# Step 1: Download stock data (e.g., Tesla stock data)
stock_symbol = 'AAPL'
data = yf.download(stock_symbol, start='2020-01-01', end='2023-12-31')

# Step 2: Feature engineering (predicting next day's close price)
data['Close'] = data['Close'].shift(-1)  # Shift to predict the next day's closing price

# Drop the last row with NaN value
data = data.dropna()

# Use 'Open', 'High', 'Low', and 'Volume' as features
X = data[['Open', 'High', 'Low', 'Volume']].values
y = data['Close'].values

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Initialize and train the Support Vector Machine (SVM) model
svm_model = SVR(kernel='rbf')
svm_model.fit(X_train, y_train)

# Step 5: Make predictions on the test set
y_pred = svm_model.predict(X_test)

# Step 6: Calculate the evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Create classification-like metrics for thresholds
thresholds = [0.01, 0.03, 0.05, 0.10]  # Different error thresholds (1%, 3%, 5%, 10%)
metrics = []

for threshold in thresholds:
    accuracy = np.mean(np.abs((y_pred - y_test) / y_test) <= threshold) * 100
    metrics.append((threshold * 100, accuracy))  # Save as percentage

# Display metrics in a table-like format
print("\nThreshold vs Accuracy Metrics")
print("{:<15} {:<15}".format("Threshold (%)", "Accuracy (%)"))
for threshold, acc in metrics:
    print("{:<15} {:<15.2f}".format(threshold, acc))

# Step 7: Plot only the last 30 days of predictions and actual values
# Select the last 30 days
last_30_days_true = y_test[-30:]
last_30_days_pred = y_pred[-30:]

plt.figure(figsize=(10, 6))
plt.plot(last_30_days_true, label='Harga Real', color='blue')
plt.plot(last_30_days_pred, label='Prediksi Harga', linestyle='--', color='red')
plt.title(f'{stock_symbol} Stock Price Prediction (Last 30 Days)')
plt.xlabel('Days')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()
