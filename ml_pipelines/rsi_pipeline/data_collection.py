# Alpaca API imports
from alpaca.data import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame


from datetime import datetime
import pandas as pd

import os
from dotenv import load_dotenv

load_dotenv()

# Get the API key from the environment variable
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")



class DataCollector:
    def __init__(self, api_key, secret_key, output_dir="stock_data", combined_output_file="combined_data.csv"):
        """
        Initialize the data collector with Alpaca API credentials and output paths.
        :param api_key: Alpaca API Key.
        :param secret_key: Alpaca Secret Key.
        :param output_dir: Directory to save individual CSV files.
        :param combined_output_file: Path to save combined data CSV.
        """
        self.api_key = api_key
        self.secret_key = secret_key
        self.output_dir = output_dir
        self.combined_output_file = combined_output_file

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize Alpaca Data Client
        self.data_client = StockHistoricalDataClient(self.api_key, self.secret_key)

    def fetch_stock_data(self, symbol, start_date, end_date, timeframe=TimeFrame.Day):
        """
        Fetch historical stock data for a specific stock symbol.
        :param symbol: Stock symbol (e.g., "AAPL").
        :param start_date: Start date for data collection (datetime object).
        :param end_date: End date for data collection (datetime object).
        :param timeframe: Timeframe for stock bars (default: TimeFrame.Day).
        :return: Pandas DataFrame with stock data or None if an error occurs.
        """
        try:
            print(f"Fetching data for {symbol}...")
            request_params = StockBarsRequest(
                symbol_or_symbols=symbol,
                start=start_date,
                end=end_date,
                timeframe=timeframe
            )

            # Fetch data and convert to DataFrame
            bars = self.data_client.get_stock_bars(request_params)
            df = bars.df.reset_index()  # Convert to DataFrame and reset index
            df['symbol'] = symbol  # Add symbol column for easier identification

            return df
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return None

    def collect_multiple_stocks(self, symbols, start_date, end_date, save_combined=True):
        """
        Collect historical stock data for multiple stocks and save results to CSV.
        :param symbols: List of stock symbols (e.g., ["AAPL", "MSFT"]).
        :param start_date: Start date for data collection (datetime object).
        :param end_date: End date for data collection (datetime object).
        :param save_combined: Whether to save all data as a single combined CSV (default: True).
        """
        all_data = []

        for symbol in symbols:
            df = self.fetch_stock_data(symbol, start_date, end_date)
            if df is not None:
                # Save individual stock data to a CSV file
                output_file = os.path.join(self.output_dir, f"{symbol}_data.csv")
                df.to_csv(output_file, index=False)
                print(f"Saved {symbol} data to {output_file}")
                all_data.append(df)

        if save_combined and all_data:
            # Combine all data and save to a single CSV file
            combined_df = pd.concat(all_data, ignore_index=True)
            combined_df.to_csv(self.combined_output_file, index=False)
            print(f"Saved combined data to {self.combined_output_file}")

        print("Data collection complete.")


# Main script
if __name__ == "__main__":
    

    # Stock symbols to fetch
    stock_symbols = ["NVDA", "AAPL", "MSFT", "AMZN", "GOOG", "META", "TSLA", "BRK.B", "TSM", "AVGO"]

    # Date range for historical data
    start_date = datetime(2023, 11, 20)
    end_date = datetime(2024, 11, 19)

    # Initialize the collector
    collector = DataCollector(
        api_key=ALPACA_API_KEY,
        secret_key=ALPACA_SECRET_KEY,
        output_dir="stock_data",
        combined_output_file="combined_stock_data.csv"
    )

    # Collect data for the specified stocks
    collector.collect_multiple_stocks(stock_symbols, start_date, end_date)
