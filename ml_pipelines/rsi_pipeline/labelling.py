import pandas as pd
import numpy as np


class Labeler:
    """
    A class to label stock data based on future percentage changes and generate sequential data.
    """

    def __init__(self, input_csv = "rsi_stock_data.csv", forecast_days=1, threshold=0.01, seq_length=50,):
        """
        Initialize the SignalLabeler with configurations.

        Parameters:
        - input_csv (str): Path to the input CSV file containing stock data.
        - forecast_days (int): Number of days ahead to predict. Default is 1.
        - threshold (float): Percentage change threshold for generating signals. Default is 0.01 (1%).
        - seq_length (int): Number of past days to consider for sequences. Default is 50.
        """
        self.input_csv = input_csv
        self.forecast_days = forecast_days
        self.threshold = threshold
        self.seq_length = seq_length
       


    def load_data(self):
        """
        Load the stock data from the input CSV file.

        Returns:
        - DataFrame: A Pandas DataFrame containing the stock data.
        """
        print(f"Loading data from {self.input_csv}...")
        data = pd.read_csv(self.input_csv, parse_dates=['timestamp'])
        return data

    def label_data(self, data):
        """
        Label stock data based on future percentage changes.

        Parameters:
        - data (DataFrame): A Pandas DataFrame containing stock data.

        Returns:
        - DataFrame: A labeled DataFrame with 'signal' and 'future_pct_change' columns.
        """
        print("Labeling data...")
        # Calculate future price change percentage
        data['future_pct_change'] = data['close'].pct_change(periods=self.forecast_days).shift(-self.forecast_days)

        # Define the label: 1 for buy, -1 for sell, 0 for hold
        data['signal'] = np.where(data['future_pct_change'] > self.threshold, 1,
                                  np.where(data['future_pct_change'] < -self.threshold, -1, 0))

        # Drop rows with NaN values
        data = data.dropna(subset=['future_pct_change', 'signal'])

        return data

    def create_sequences(self, X, y):
        """
        Create sequences of features and corresponding labels.

        Parameters:
        - X (DataFrame): Feature data.
        - y (Series): Labels corresponding to the data.

        Returns:
        - Tuple: Arrays of sequences (X_sequences) and labels (y_labels).
        """
        print("Creating sequences...")
        xs, ys = [], []
        for i in range(len(X) - self.seq_length):
            x_seq = X.iloc[i:(i + self.seq_length)].values
            y_label = y.iloc[i + self.seq_length]
            xs.append(x_seq)
            ys.append(y_label)
        return np.array(xs), np.array(ys)

    def process(self):
        """
        Process the stock data to generate labeled sequences.

        Returns:
        - Tuple: Arrays of feature sequences (X_sequences) and labels (y_labels).
        """
        # Load data
        data = self.load_data()

        # Label the data
        data_with_labels = self.label_data(data)

        # Drop columns not needed for features
        X = data_with_labels.drop(columns=['future_pct_change', 'signal'])
        y = data_with_labels['signal']

        # Create sequences
        X_sequences, y_labels = self.create_sequences(X, y)

        print("Data processing complete.")
        return X_sequences, y_labels
