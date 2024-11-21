import pandas as pd
import numpy as np

# Load the stock data
data = pd.read_csv('rsi_stock_data.csv', parse_dates=['timestamp'])

forecast_days = 1  # Number of days ahead to predict
threshold = 0.01   # 1% change threshold

def Labels(data, forecast_days, threshold):
    # Calculate future price change percentage
    data['future_pct_change'] = data['close'].pct_change(periods=forecast_days).shift(-forecast_days)

    # Define the label: 1 for buy, -1 for sell, 0 for hold
    data['signal'] = np.where(data['future_pct_change'] > threshold, 1,
                              np.where(data['future_pct_change'] < -threshold, -1, 0))

    # Drop rows with NaN values
    data = data.dropna(subset=['future_pct_change', 'signal'])

    return data

# Generate labels
data_with_labels = Labels(data, forecast_days, threshold)

# Drop columns not needed for features
X = data_with_labels.drop(columns=['future_pct_change', 'signal'])
y = data_with_labels['signal']

def create_sequences(X, y, seq_length):
    xs, ys = [], []
    for i in range(len(X) - seq_length):
        x_seq = X.iloc[i:(i + seq_length)].values
        y_label = y.iloc[i + seq_length]
        xs.append(x_seq)
        ys.append(y_label)
    return np.array(xs), np.array(ys)

seq_length = 50  # Number of past days to consider
X_sequences, y_labels = create_sequences(X, y, seq_length)
