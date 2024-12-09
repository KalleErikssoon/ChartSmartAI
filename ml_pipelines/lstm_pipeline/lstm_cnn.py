from alpaca.data import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from sklearn.preprocessing import MinMaxScaler

import os
from dotenv import load_dotenv

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

import torch
import torch.nn as NN
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

# Load environment variables
load_dotenv()

# Retrieve API keys
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

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
print("Original data preview:")
print(data.head())

# Preprocess data: Select necessary columns
data = data[['close']]
data.reset_index(inplace=True)  # Convert index to a column
data = data[['timestamp', 'close']]
data['timestamp'] = pd.to_datetime(data['timestamp'])

# Normalize data
scaler = MinMaxScaler()
scaled_data = pd.DataFrame(
    scaler.fit_transform(data[['close']]),
    columns=['close'],
    index=data.timestamp
)

print("Scaled data preview:")
print(scaled_data.head())

# Function to create sequences for time series
def create_sequences(data, window_size, prediction_horizon):
    X, y = [], []
    for i in range(len(data) - window_size - prediction_horizon + 1):
        X.append(data.iloc[i:i + window_size].values)
        y.append(data.iloc[i + window_size:i + window_size + prediction_horizon]['close'].values)
    X = np.array(X)
    y = np.array(y)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# Parameters for data preparation
window_size = 30
prediction_horizon = 3

# Create sequences
X, y = create_sequences(scaled_data, window_size, prediction_horizon)

# Split the dataset into training and testing
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Convert to DataLoader
batch_size = 32
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

# Define CNN + LSTM Model
class CNN_LSTM_Model(NN.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, cnn_channels, kernel_size):
        super(CNN_LSTM_Model, self).__init__()
        
        # CNN layer
        self.conv1 = NN.Conv1d(in_channels=input_size, out_channels=cnn_channels, kernel_size=kernel_size, stride=1)
        self.pool = NN.MaxPool1d(kernel_size=2, stride=2)  # Pooling layer
        
        # LSTM layer
        self.lstm = NN.LSTM(input_size=cnn_channels, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        
        # Fully connected layer
        self.fc = NN.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # Input shape: (batch_size, seq_len, input_size)
        
        # Reshape for CNN: (batch_size, input_size, seq_len)
        x = x.permute(0, 2, 1)
        
        # Pass through CNN layers
        x = self.conv1(x)  # Convolutional layer
        x = F.relu(x)  # Activation function
        x = self.pool(x)  # Max pooling
        
        # Reshape for LSTM: (batch_size, seq_len, cnn_channels)
        x = x.permute(0, 2, 1)
        
        # Pass through LSTM
        out, (h_n, c_n) = self.lstm(x)  # LSTM layer
        
        # Fully connected layer
        out = self.fc(out[:, -1, :])  # Only use the last time step
        
        return out

# Model initialization
model = CNN_LSTM_Model(
    input_size=1, hidden_size=100, num_layers=2, 
    output_size=prediction_horizon, cnn_channels=32, kernel_size=3
)

# Define loss and optimizer
criterion = NN.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 30
for epoch in range(num_epochs):
    model.train()
    for batch_X, batch_y in train_loader:
        #batch_X = batch_X.unsqueeze(-1)  # Ensure input shape
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

# Evaluation
model.eval()
with torch.no_grad():
    predictions = []
    actual = []
    for batch_X, batch_y in test_loader:
        #batch_X = batch_X.unsqueeze(-1)  # Ensure input shape
        outputs = model(batch_X)
        predictions.append(outputs.numpy())
        actual.append(batch_y.numpy())

# Convert predictions and actual to arrays
predictions = np.concatenate(predictions, axis=0)
actual = np.concatenate(actual, axis=0)

# Rescale predictions back to original scale
predicted_prices = scaler.inverse_transform(predictions)
actual_prices = scaler.inverse_transform(actual)

# Calculate RMSE
rmse = mean_squared_error(actual_prices, predicted_prices, squared=False)
print(f"Test RMSE: {rmse:.4f}")

# Plot results
plt.figure(figsize=(14, 8))
for day in range(prediction_horizon):
    plt.plot(
        range(len(actual_prices)),
        actual_prices[:, day],
        label=f'Actual Day {day+1}',
        linestyle='-'
    )
    plt.plot(
        range(len(predicted_prices)),
        predicted_prices[:, day],
        label=f'Predicted Day {day+1}',
        linestyle='--'
    )
plt.title('AAPL Stock Price Prediction - CNN + LSTM')
plt.xlabel('Time')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid()
plt.show()
