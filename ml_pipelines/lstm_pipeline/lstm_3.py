# This is a script that trains a LSTM model  to prdiction 3days data and saves the model to a file

from alpaca.data import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from sklearn.preprocessing import MinMaxScaler

import os
from dotenv import load_dotenv

import pandas as pd
from datetime import datetime
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import torch.nn as NN
import torch
from torch.utils.data import DataLoader, TensorDataset

# Load environment variables
load_dotenv()

# Retrieve API keys
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
# FILE_PATH = os.getenv("FILE_PATH")  # Optional: Path to save data

# Initialize Alpaca client
client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)

# Request historical data
request_params = StockBarsRequest(
    symbol_or_symbols=["AAPL"],  # Example: Apple stock
    timeframe=TimeFrame.Day,
    start=datetime(2024, 1, 1),
    end=datetime(2024, 12, 1)
)

# Fetch data
data = client.get_stock_bars(request_params).df
print(data.head())

# Preprocess data: Select necessary columns
data = data[['close']]
data.reset_index(inplace=True)  # Convert index to a column
print(data.head())


# Ensure only numerical columns are selected
data = data[['timestamp', 'close']]  # If only 'close' is used
print("Filtered data columns:", data.columns)

# Check data types
print(data.dtypes)

# Convert non-numerical types to floats (if necessary)
data['close'] = pd.to_numeric(data['close'], errors='coerce')

# Drop missing values
data = data.dropna()
timestamps = data['timestamp']

# Normalize data
scaler = MinMaxScaler()
scaled_data = pd.DataFrame(
    scaler.fit_transform(data[['close']]),
    columns=['close'],
    index=data.timestamp
)

print("Scaled data preview:")
print(scaled_data.head())

# Function to create sequences for LSTM
def create_sequences(data, window_size, prediction_horizon):
    X, y = [], []
    for i in range(len(data) - window_size - prediction_horizon + 1):
        X.append(data.iloc[i:i + window_size].values)
        y.append(data.iloc[i + window_size:i + window_size + prediction_horizon]['close'].values)
    X = np.array(X)
    y = np.array(y)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# Parameters
window_size = 30
prediction_horizon = 3
X, y = create_sequences(scaled_data, window_size, prediction_horizon)

# Define LSTM model
class StockPriceLSTM(NN.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(StockPriceLSTM, self).__init__()
        self.lstm = NN.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = NN.Linear(hidden_size, output_size)
    
    def forward(self, x):
       # _, (h_n, _) = self.lstm(x)
       # out = self.fc(h_n[-1])
        out, (h_n, c_n) = self.lstm(x)  # Unpack outputs
        out = self.fc(out[:, -1, :])  # Output only the last time step
        return out

# Model parameters
input_size = 1  # Number of features: close
hidden_size = 100
num_layers = 2
output_size = prediction_horizon

model = StockPriceLSTM(input_size, hidden_size, num_layers, output_size)

# Define loss function and optimizer
criterion = NN.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Split the dataset into training and testing
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Convert to DataLoader
batch_size = 32
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

# Training loop
num_epochs = 30
for epoch in range(num_epochs):
    model.train()
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

# Evaluate the model
model.eval()
with torch.no_grad():
    predictions = model(X_test).numpy()
    actual = y_test.numpy()

# Inverse transform to get actual prices
predicted_prices = scaler.inverse_transform(predictions)
actual_prices = scaler.inverse_transform(actual)


last_input_window = data['timestamp'][-window_size:]  
print("The time range corresponding to the last input window：")
print(last_input_window)

# Get the corresponding time predicted for the last 3 days of the model
last_prediction_dates = data['timestamp'][-prediction_horizon:]  # The corresponding time predicted for the last 3 days of the model
print("The corresponding time predicted for the last 3 days of the model：")
print(last_prediction_dates)


# Visualization
time_index = data['timestamp'][-len(actual_prices):]

# Function to classify changes
def classify_changes(predicted_prices):
    return np.where(predicted_prices > 0, 1, 0)  # 1: Up, 0: Down
def classify_changes(predicted_prices, threshold=0.01):
    return np.where(
        predicted_prices > threshold, 1,  # Up
        np.where(predicted_prices < -threshold, -1, 0)  # Down, Stable
    )

# Convert predictions and actual prices to classifications
actual_changes = classify_changes(np.diff(actual_prices, axis=0).flatten())  # Convert actual prices to changes
predicted_changes = classify_changes(np.diff(predicted_prices, axis=0).flatten())  # Convert predictions to changes

# Ensure the arrays are 1D
print("Actual changes shape:", actual_changes.shape)
print("Predicted changes shape:", predicted_changes.shape)

# Calculate evaluation metrics
accuracy = accuracy_score(actual_changes, predicted_changes)
precision = precision_score(actual_changes, predicted_changes, average='weighted')
recall = recall_score(actual_changes, predicted_changes, average='weighted')
f1 = f1_score(actual_changes, predicted_changes, average='weighted')

# Print metrics
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")
print(f"F1 Score: {f1 * 100:.2f}%")

from sklearn.metrics import mean_squared_error, mean_absolute_error
#Evaluate the accuracy of price forecasts using regression metrics such as RMSE, MAEv
rmse = mean_squared_error(actual_prices.flatten(), predicted_prices.flatten(), squared=False)
mae = mean_absolute_error(actual_prices.flatten(), predicted_prices.flatten())
print(f"RMSE: {rmse:.2f}, MAE: {mae:.2f}")

plt.figure(figsize=(14, 8))
for day in range(predicted_prices.shape[1]):  # Iterate over each predicted day
    plt.plot(
        time_index,
        actual_prices[:, day],
        label=f'Actual Day {day+1}',
        linestyle='-'
    )
    plt.plot(
        time_index,
        predicted_prices[:, day],
        label=f'Predicted Day {day+1}',
        linestyle='--'
    )

plt.title('AAPL Stock Price Prediction - LSTM', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Close Price (USD)', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('LAST_3day01_prediction.png', dpi=300, bbox_inches='tight')
plt.show()
