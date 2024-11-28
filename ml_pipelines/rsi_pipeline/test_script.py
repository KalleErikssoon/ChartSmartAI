from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

from datetime import datetime
import os
import django
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from django.utils import timezone

# Setting Django environment variable
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'django_ai_test_project.settings')
django.setup()
from stockAPP.models import APPLbar

# Initialize Alpaca stock data client
client = StockHistoricalDataClient("ALPACA_API_KEY", "ALPACA_SECRET_KEY")

# Delete all existing data in the table
APPLbar.objects.all().delete()

# Request parameters
request_params = StockBarsRequest(
    symbol_or_symbols="AAPL",
    timeframe=TimeFrame.Day,
    start=datetime(2024, 1, 1),
    end=datetime(2024, 11, 11)
)

# Fetching bar data
bars = client.get_stock_bars(request_params)

for bar in bars["AAPL"]:
    naive_timestamp = bar.timestamp
    if timezone.is_naive(naive_timestamp):
        aware_timestamp = timezone.make_aware(naive_timestamp, timezone.get_current_timezone())
    else:
        aware_timestamp = naive_timestamp

    APPLbar.objects.update_or_create(
        timestamp=aware_timestamp,
        defaults={
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "volume": bar.volume,
        },
    )
print("...Data has been saved to the SQLite database...")

# Query data from the database
queryset = APPLbar.objects.all()
data = list(queryset.values())

# Convert the data to a DataFrame
df = pd.DataFrame(data)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.set_index('timestamp', inplace=True)

# Handle missing values
df.fillna(method='bfill', inplace=True)

# Feature selection
df = df[['open', 'high', 'low', 'close', 'volume']]  # Using 'close' for prediction

# Normalize the features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df)

# **2. Create time series data**
def create_sequences(data, sequence_length):
    x, y = [], []
    for i in range(len(data) - sequence_length):
        x.append(data[i:i+sequence_length])  # Use past `sequence_length` days as input
        y.append(data[i+sequence_length, 3])  # Predict the `close` value of day `sequence_length+1`
    return np.array(x), np.array(y)

sequence_length = 50
x, y = create_sequences(scaled_data, sequence_length)

# Split the data into training and testing sets (80% training, 20% testing)
train_size = int(len(x) * 0.8)
x_train, x_test = x[:train_size], x[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# **3. Build and train the LSTM model**
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=50, batch_size=32, verbose=1)

# **4. # Predict and inverse transform
y_pred = model.predict(x_test)
full_shape = np.zeros((len(y_pred), scaled_data.shape[1]))
full_shape[:, 3] = y_pred.flatten()
inversed_data = scaler.inverse_transform(full_shape)
predicted_close = inversed_data[:, 3]

full_shape_test = np.zeros((len(y_test), scaled_data.shape[1]))
full_shape_test[:, 3] = y_test.flatten()  
y_test_actual = scaler.inverse_transform(full_shape_test)[:, 3]

# plot
plt.figure(figsize=(14, 8))
plt.plot(df.index[-len(y_test_actual):], y_test_actual, label='Actual Price', color='blue', linewidth=2)
plt.plot(df.index[-len(predicted_close):], predicted_close, label='Predicted Price', color='orange', linestyle='--', linewidth=2)
plt.title('AAPL Stock Price Prediction Using LSTM', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Close Price (USD)', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig('AAPL_stock_prediction.png', dpi=300, bbox_inches='tight')
plt.show()



"""
#  mplfinance Drawing K-plots
mpf.plot(
    df,
    type='candle',         
    volume=True,           
    title="AAPL Daily Candlestick Chart",
    ylabel='Price (USD)',
    ylabel_lower='Volume', 
    style='yahoo',         
    datetime_format='%Y-%m',  
    xrotation=20,
    savefig ='AAPL_Candlestick.png'
)

# Plot a time series of volumefigure(figsize=(12, 6))
plt.bar(df.index, df['volume'], color="red", label="Volume")
plt.title("AAPL Daily Trading Volume")
plt.xlabel("Date")
plt.ylabel("Volume")
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{output_folder}/AAPL_Volume.png")
plt.show()

# Calculate the 10-day and 30-day simple moving averages (SMA)
df['SMA_10'] = df['close'].rolling(window=10).mean()
df['SMA_30'] = df['close'].rolling(window=30).mean()

# Exponential Moving Average (EMA)
df['EMA_10'] = df['close'].ewm(span=10, adjust=False).mean()
df['EMA_30'] = df['close'].ewm(span=30, adjust=False).mean()

# Plot the closing price and SMA,EMA
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['close'], label="Close Price", linewidth=1)
plt.plot(df.index, df['SMA_10'], label="10-Day SMA", linewidth=1, color='orange',linestyle='--')
plt.plot(df.index, df['SMA_30'], label="30-Day SMA", linewidth=1, color='red',linestyle='--')
plt.plot(df.index, df['EMA_10'], label="10-Day EMA", linewidth=1, color='green')
plt.plot(df.index, df['EMA_30'], label="30-Day EMA", linewidth=1, color='purple')
plt.title("AAPL Close Price with Moving Averages SMAs and EMAs")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.grid(True)
plt.savefig(f"{output_folder}/AAPL_SMAs_EMAs.png")
plt.show()

#  RSI
def calculate_rsi(data, window=14):
    delta = data['close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

df['RSI'] = calculate_rsi(df)

#  MACD
df['EMA_12'] = df['close'].ewm(span=12, adjust=False).mean()
df['EMA_26'] = df['close'].ewm(span=26, adjust=False).mean()
df['MACD'] = df['EMA_12'] - df['EMA_26']
df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

#  RSI 
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['RSI'], label="RSI", color='blue')
plt.axhline(70, linestyle='--', color='red', label='Overbought (70)')
plt.axhline(30, linestyle='--', color='green', label='Oversold (30)')
plt.title("AAPL Relative Strength Index (RSI)")
plt.xlabel("Date")
plt.ylabel("RSI")
plt.legend()
plt.grid(True)
plt.savefig(f"{output_folder}/AAPL_RSI.png")
plt.show()

# draw MACD indicator and signal line
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['MACD'], label="MACD", color='purple')
plt.plot(df.index, df['Signal_Line'], label="Signal Line", color='orange')
plt.bar(df.index, df['MACD'] - df['Signal_Line'], color='gray', alpha=0.3, label="MACD Histogram")
plt.title("AAPL Moving Average Convergence Divergence (MACD)")
plt.xlabel("Date")
plt.ylabel("MACD")
plt.legend()
plt.grid(True)
plt.savefig(f"{output_folder}/AAPL_MACD.png")
plt.show()


print(df[['SMA_10', 'SMA_30', 'EMA_10', 'EMA_30', 'RSI', 'MACD', 'Signal_Line']].tail())

# close database Connection
conn.close()

"""