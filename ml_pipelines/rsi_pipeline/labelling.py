import pandas as pd

class Labelling:
    """
    A class to generate buy/hold/sell labels based on RSI, close price, and volume.
    """

    def __init__(self, input_csv="rsi_stock_data.csv", output_csv="rsi_stock_data.csv"):
        """
        Initialize the LabellingProcessor with file paths.

        Parameters:
        - input_csv (str): Path to the input CSV file containing stock data with RSI.
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

    def labels(self, data, rsi_buy_threshold=30, rsi_sell_threshold=70, volume_window=5):
        """
        Generate buy/hold/sell labels based on RSI, close price, and volume.

        Parameters:
        - data (DataFrame): A Pandas DataFrame containing the stock data with RSI values.
        - rsi_buy_threshold (float): RSI threshold to trigger a buy signal.
        - rsi_sell_threshold (float): RSI threshold to trigger a sell signal.
        - volume_window (int): Rolling window size for volume comparison.

        Returns:
        - DataFrame: A DataFrame with an additional 'label' column.
        """
        print("Generating labels based on RSI, close price, and volume...")

        # Ensure required columns exist
        required_columns = ['RSI', 'close', 'volume']
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Missing required column: {col}")

        # Calculate rolling average for volume
        data['volume_avg'] = data['volume'].rolling(volume_window).mean()

        # Initialize labels
        data['label'] = 0  # Default to hold

        # Buy signals
        data.loc[
            (data['RSI'] < rsi_buy_threshold) & 
            (data['volume'] > data['volume_avg']),
            'label'
        ] = 1

        # Sell signals
        data.loc[
            (data['RSI'] > rsi_sell_threshold) & 
            (data['volume'] > data['volume_avg']),
            'label'
        ] = -1

        # Drop temporary columns if not needed
        data.drop(columns=['volume_avg'], inplace=True)

        return data

    def save_data(self, data):
        """
        Save the labeled stock data to the output CSV file.

        Parameters:
        - data (DataFrame): A Pandas DataFrame containing the labeled stock data.
        """
        print(f"Saving labeled data to {self.output_csv}...")
        data.to_csv(self.output_csv, index=False)

    def process(self, rsi_buy_threshold=30, rsi_sell_threshold=70, volume_window=5):
        """
        Main method to generate labels and save the results.

        Parameters:
        - rsi_buy_threshold (float): RSI threshold to trigger a buy signal.
        - rsi_sell_threshold (float): RSI threshold to trigger a sell signal.
        - volume_window (int): Rolling window size for volume comparison.
        """
        # Load the data
        data = self.load_data()

        # Generate labels
        labeled_data = self.labels(data, rsi_buy_threshold, rsi_sell_threshold, volume_window)

        # Save the labeled data
        self.save_data(labeled_data)

        print("Labelling process complete.")
