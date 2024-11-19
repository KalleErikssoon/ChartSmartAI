# Import necessary libraries
# Import necessary libraries
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
import ta  # TA-Lib for technical indicators

# 设置 Django 环境变量
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'django_ai_test_project.settings')
django.setup()
from stockAPP.models import APPLbar

# Initialize the Alpaca stock data client
client = StockHistoricalDataClient("YOUR-API-KEY", "SECRET-KEY")

# Setting request parameters
request_params = StockBarsRequest(
    symbol_or_symbols="AAPL",
    timeframe=TimeFrame.Day,
    start=datetime(2023, 1, 1),
    end=datetime(2024, 4, 30)
)

# Getting bar data
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

# Query data
queryset = APPLbar.objects.all()
data = list(queryset.values())

# Convert the data to a DataFrame
df = pd.DataFrame(data)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.set_index('timestamp', inplace=True)

# 添加技术指标
df['SMA_20'] = ta.trend.sma_indicator(df['close'], window=20)  # 20日均线
df['MACD'] = ta.trend.macd(df['close'])  # MACD 指标
df['RSI'] = ta.momentum.rsi(df['close'], window=14)  # 相对强弱指标 RSI

# 删除 NaN 值
df.fillna(method='bfill', inplace=True)

# 归一化所有特征
feature_columns = ['open', 'high', 'low', 'close', 'volume', 'SMA_20', 'MACD', 'RSI']
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df[feature_columns].values)

# 创建时间序列
def create_sequences(data, sequence_length):
    x, y = [], []
    for i in range(len(data) - sequence_length):
        x.append(data[i:i+sequence_length])
        y.append(data[i+sequence_length, 3])  # 预测的是 'close' 收盘价（特征列表中的索引 3）
    return np.array(x), np.array(y)

sequence_length = 50
x, y = create_sequences(scaled_data, sequence_length)

# 划分训练集和测试集
split_ratio = 0.8
split_index = int(len(x) * split_ratio)
x_train, x_test = x[:split_index], x[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# 构建 LSTM 模型
model = Sequential([
    LSTM(200, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])),  # 增加神经元数量
    Dropout(0.4),
    LSTM(200, return_sequences=False),
    Dropout(0.4),
    Dense(100),
    Dense(1)
])

# 编译模型
model.compile(optimizer='adam', loss='mean_squared_error')

# 使用 EarlyStopping 防止过拟合
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=10)

# 增加训练轮数
model.fit(x_train, y_train, batch_size=32, epochs=100, callbacks=[early_stopping])

# 预测测试集数据
predicted_prices = model.predict(x_test)

# 只反归一化 'close' 列（特征列表中的索引 3）
scaler_partial = MinMaxScaler()
scaler_partial.min_, scaler_partial.scale_ = scaler.min_[3], scaler.scale_[3]
predicted_prices = scaler_partial.inverse_transform(predicted_prices.reshape(-1, 1))

# 反归一化 y_test
y_test_inverse = scaler_partial.inverse_transform(y_test.reshape(-1, 1))

# 绘制预测结果
plt.figure(figsize=(14, 8))
plt.plot(df.index[-len(y_test_inverse):], y_test_inverse, label="Actual Price", color='blue')
plt.plot(df.index[-len(predicted_prices):], predicted_prices, label="Predicted Price", color='orange')
plt.title("AAPL Stock Price Prediction")
plt.xlabel("Date")
plt.ylabel("Close Price (USD)")
plt.legend()
plt.grid(True)
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