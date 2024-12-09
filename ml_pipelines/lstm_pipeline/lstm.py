# This is a script that trains a LSTM model on the data and saves the model to a file

# This is a script that trains an LSTM model on the data and saves the model to a file

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

import torch.nn as NN
import torch

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
def create_sequences(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data.iloc[i:i + window_size].values)  # Data within the window
        y.append(data.iloc[i + window_size]['close'])  # Target value after the window
    X = np.array(X)
    y = np.array(y)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

window_size = 60
X, y = create_sequences(scaled_data, window_size)
print(f"Input shape: {X.shape}, Output shape: {y.shape}")

# Define LSTM model
class StockPriceLSTM(NN.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(StockPriceLSTM, self).__init__()
        self.lstm = NN.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = NN.Linear(hidden_size, output_size)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        out = self.fc(h_n[-1])  # Use the last hidden state
        return out

# Model parameters
input_size = 1  # Number of features: close
hidden_size = 50
num_layers = 2
output_size = 1

model = StockPriceLSTM(input_size, hidden_size, num_layers, output_size)
print(model)

# Define loss function and optimizer
criterion = NN.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Split the dataset into training and testing
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Convert to DataLoader
from torch.utils.data import DataLoader, TensorDataset

batch_size = 32
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

# Training loop
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs.squeeze(), batch_y)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

# Evaluate the model
model.eval()
with torch.no_grad():
    predictions = model(X_test).squeeze().numpy()
    actual = y_test.numpy()

# Inverse transform to get actual prices
predicted_prices = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
actual_prices = scaler.inverse_transform(actual.reshape(-1, 1)).flatten()

# Set time series as x-axis
time_index = data['timestamp'][-len(actual_prices):]

plt.figure(figsize=(14, 8))

# Plot actual prices
plt.plot(time_index, actual_prices, label='Actual Prices', color='blue', linewidth=2)

# Plot predicted prices
plt.plot(time_index, predicted_prices, label='Predicted Prices', color='orange', linestyle='--', linewidth=2)

# Add title and labels
plt.title('AAPL Stock Price Prediction Using LSTM', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Close Price (USD)', fontsize=14)

# Add legend and grid
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)

# Show the plot
plt.show()


