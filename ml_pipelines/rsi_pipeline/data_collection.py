import os
from datetime import datetime
import pandas as pd
from alpaca.data import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from dotenv import load_dotenv
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Load environment variables
load_dotenv()

FILE_PATH = os.getenv('RSI_FILE_PATH')



class DataCollector:
    def __init__(self, api_key=None, secret_key=None, output_path=FILE_PATH):
        print("Initializing DataCollector...")
        self.api_key = api_key or os.getenv('ALPACA_API_KEY')
        self.secret_key = secret_key or os.getenv('ALPACA_SECRET_KEY')
        self.output_path = output_path

        if not self.api_key or not self.secret_key:
            raise ValueError("Alpaca API Key and Secret Key must be provided.")

        # Initialize the Alpaca Data Client
        self.data_client = StockHistoricalDataClient(self.api_key, self.secret_key)
        print("DataCollector initialized successfully.")

    def collect_data(self):
        print("Starting data collection...")
        stocks = ["NVDA", "AAPL", "MSFT", "AMZN", "GOOG", "META", "TSLA", "BRK.B", "TSM", "AVGO"]
        stock_data = []

        for symbol in stocks:
            print(f"Collecting data for {symbol}...")
            try:
                request_params = StockBarsRequest(
                    symbol_or_symbols=symbol,
                    start=datetime(2023, 11, 20),
                    end=datetime(2024, 11, 19),
                    timeframe=TimeFrame.Day
                )
                bars = self.data_client.get_stock_bars(request_params)
                df = bars.df.reset_index()
                df['symbol'] = symbol
                stock_data.append(df)
                print(f"Data for {symbol} collected successfully.")
            except Exception as e:
                print(f"Failed to collect data for {symbol}: {e}")

        if stock_data:
            final_df = pd.concat(stock_data, ignore_index=True)
            final_df.to_csv(self.output_path, index=False)
            print(f"Successfully collected data and saved to {self.output_path}")
        else:
            print("No data collected.")

