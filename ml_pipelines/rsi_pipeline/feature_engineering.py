import pandas as pd
from ta.momentum import RSIIndicator
from sklearn.preprocessing import MinMaxScaler

def process_stock_data(input_csv, output_csv, rsi_period=14):
    """
    Processes stock data by calculating the RSI and normalizing specified features.

    Parameters:
    - input_csv (str): Path to the input CSV file containing stock data.
    - output_csv (str): Path to save the processed CSV file.
    - rsi_period (int): Period for RSI calculation. Default is 14.

    Returns:
    - None. The processed data is saved to the specified output CSV file.
    """
    # Load the stock data
    data = pd.read_csv(input_csv, parse_dates=['timestamp'])

    # Initialize an empty list to store DataFrames
    rsi_dataframes = []

    # Group by 'symbol' and calculate RSI for each group
    grouped = data.groupby('symbol')
    for symbol, group in grouped:
        # Ensure the group is sorted by 'timestamp'
        group = group.sort_values('timestamp')

        # Calculate RSI
        rsi_indicator = RSIIndicator(close=group['close'], window=rsi_period)
        group['RSI'] = rsi_indicator.rsi()

        # Append the DataFrame to the list
        rsi_dataframes.append(group)

    # Concatenate all DataFrames
    data_with_rsi = pd.concat(rsi_dataframes)

    # Drop rows with NaN values in the 'RSI' column
    data_with_rsi = data_with_rsi.dropna(subset=['RSI'])

    # Select features to normalize
    features_to_normalize = ['open', 'high', 'low', 'close', 'volume', 'trade_count', 'vwap', 'RSI']

    # Initialize the scaler
    scaler = MinMaxScaler()

    # Apply normalization
    data_with_rsi[features_to_normalize] = scaler.fit_transform(data_with_rsi[features_to_normalize])

    # Save the DataFrame to a new CSV file
    data_with_rsi.to_csv(output_csv, index=False)
