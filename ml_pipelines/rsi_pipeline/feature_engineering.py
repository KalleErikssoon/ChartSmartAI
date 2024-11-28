import pandas as pd
from ta.momentum import RSIIndicator
from sklearn.preprocessing import MinMaxScaler
import numpy as np


class StockDataProcessor:
    """
    A class to process stock data, including calculating the RSI and normalizing specified features.
    """

    def __init__(self, input_csv= "rsi_stock_data.csv", output_csv = "rsi_stock_data.csv", rsi_period=14):
        """
        Initialize the StockDataProcessor with file paths and configuration.

        Parameters:
        - input_csv (str): Path to the input CSV file containing stock data.
        - output_csv (str): Path to save the processed CSV file.
        - rsi_period (int): Period for RSI calculation. Default is 14.
        """
        self.input_csv = input_csv
        self.output_csv = output_csv
        self.rsi_period = rsi_period

    def load_data(self):
        """
        Load stock data from the input CSV file.

        Returns:
        - DataFrame: A Pandas DataFrame containing the stock data.
        """
        print(f"Loading data from {self.input_csv}...")
        return pd.read_csv(self.input_csv, parse_dates=['timestamp'])

    def calculate_rsi(self, data):
        """
        Calculate the RSI for the stock data grouped by symbol.

        Parameters:
        - data (DataFrame): A Pandas DataFrame containing the stock data.

        Returns:
        - DataFrame: A DataFrame with RSI values calculated.
        """
        print("Calculating RSI...")
        rsi_dataframes = []

        grouped = data.groupby('symbol')
        for symbol, group in grouped:
            group = group.sort_values('timestamp')
            rsi_indicator = RSIIndicator(close=group['close'], window=self.rsi_period)
            group['RSI'] = rsi_indicator.rsi()
            rsi_dataframes.append(group)

        return pd.concat(rsi_dataframes)
    
    def save_data(self, data):
        """
        Save the processed stock data to the output CSV file.

        Parameters:
        - data (DataFrame): A Pandas DataFrame containing the processed stock data.
        """
        print(f"Saving processed data to {self.output_csv}...")
        data.to_csv(self.output_csv, index=False)

    def process(self):
        """
        Main method to process stock data by calculating RSI and normalizing features.
        """
        # Load the data
        data = self.load_data()

        # Calculate RSI
        data_with_rsi = self.calculate_rsi(data)

        # Normalize features
        #normalized_data = self.normalize_features(data_with_rsi)

        # Save the processed data
        self.save_data(data_with_rsi)

        print("Stock data processing complete.")
