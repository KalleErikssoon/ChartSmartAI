import pandas as pd
import os

FILE_PATH = os.getenv('RSI_FILE_PATH')


class Labelling:
    """
    A class to generate buy/hold/sell labels based on future price prediction.
    """

    # def __init__(self, input_csv="rsi_stock_data.csv", output_csv="rsi_stock_data.csv"):
    #     """
    #     Initialize the LabellingProcessor with file paths.

    #     Parameters:
    #     - input_csv (str): Path to the input CSV file containing stock data.
    #     - output_csv (str): Path to save the labeled CSV file.
    #     """
    #     self.input_csv = input_csv
    #     self.output_csv = output_csv

    def __init__(self, input_csv=FILE_PATH, output_csv=FILE_PATH):
        """
        Initialize the LabellingProcessor with file paths.

        Parameters:
        - input_csv (str): Path to the input CSV file containing stock data.
        - output_csv (str): Path to save the labeled CSV file.
        """
        self.input_csv = input_csv
        self.output_csv = output_csv

    def load_data(self):
        """
        Load stock data from the input CSV file.

        Returns:
        - DataFrame: A Pandas DataFrame containing the stock data.
        """
        print(f"Loading data from {self.input_csv}...")
        return pd.read_csv(self.input_csv)

    def labels(self, data, prediction_window=3, threshold=0.01):
        """
        Generate buy/hold/sell labels based on future price prediction.

        Parameters:
        - data (DataFrame): A Pandas DataFrame containing the stock data.
        - prediction_window (int): Number of days to look ahead for predicting price change.
        - threshold (float): Minimum price change percentage to trigger buy or sell signal.

        Returns:
        - DataFrame: A DataFrame with an additional 'label' column.
        """
        print("Generating labels based on future price and threshold...")

        # Ensure required column 'close' exists
        if 'close' not in data.columns:
            raise ValueError("Missing required column: 'close'")

        # Initialize labels
        data['label'] = 1  # Default to hold (0 for buy, 1 for hold, 2 for sell, NA for no label)

        # Group data by symbol and iterate through each group
        grouped = data.groupby('symbol')

        for symbol, group in grouped:
            print(f"Processing symbol: {symbol}")
            
            # Iterate through the group (data for a specific symbol)
            for i in range(len(group) - prediction_window):
                # Get the current close price
                current_close = group['close'].iloc[i]

                # Get the future close price (after 'prediction_window' days)
                future_close = group['close'].iloc[i + prediction_window]

                # Calculate the percentage change between current and future close price
                price_change = (future_close - current_close) / current_close

                # Assign labels based on price change
                if price_change >= threshold:
                    data.at[group.index[i], 'label'] = 0  # Buy signal
                elif price_change <= -threshold:
                    data.at[group.index[i], 'label'] = 2  # Sell signal
                else:
                    data.at[group.index[i], 'label'] = 1  # Hold signal

        return data
    
    #Remove rows with missing label or rsi
    def data_cleaning(self, data):
        data.dropna(subset=['label', 'rsi'], inplace=True)
        print("Removed values with missing label or RSI value.")

    def save_data(self, data):
        """
        Save the labeled stock data to the output CSV file.

        Parameters:
        - data (DataFrame): A Pandas DataFrame containing the labeled stock data.
        """
        print(f"Saving labeled data to {self.output_csv}...")
        data.to_csv(self.output_csv, index=False)

    def process(self):
        """
        Main method to generate labels and save the results.
        """
        # Load the data
        data = self.load_data()

        # Generate labels
        labeled_data = self.labels(data)

        # REmove empty values from csv file
        self.data_cleaning(labeled_data)

        # Save the labeled data
        self.save_data(labeled_data)

        print("Labelling process complete.")
