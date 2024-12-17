
# Author: Karl Eriksson, Isaac Lindegren Ternbom
# Alpaca API imports
from alpaca.data import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

# For .env (API keys etc)
import os
from dotenv import load_dotenv

# Additional imports
from datetime import datetime, timedelta  # for stock request_params
from dateutil.relativedelta import relativedelta
import pandas as pd  # To handle data storage

# Load environment variables
load_dotenv()  # Load the .env file
ALPACA_API_KEY = os.getenv('ALPACA_API_KEY')
ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')
FILE_PATH = os.getenv('MACD_FILE_PATH')


class DataCollector:
    def __init__(self, api_key=ALPACA_API_KEY, secret_key=ALPACA_SECRET_KEY, output_path=FILE_PATH):
        """
        Initialize the DataCollector with Alpaca API credentials and output file path.
        """
        self.api_key = api_key
        self.secret_key = secret_key
        self.output_path = output_path

        # Connect to Alpaca API
        self.data_client = StockHistoricalDataClient(self.api_key, self.secret_key)

    """
    Collect stock data for the top-10 stocks from the Alpaca API and save it to a CSV file.
    """
    def collect_data(self):

        # Array of top-10 stocks
        stocks = ["NVDA", "AAPL", "MSFT", "AMZN", "GOOG", "META", "TSLA", "BRK.B", "TSM", "AVGO"]

        # List to store all collected data
        stock_data = []

        # calculate yesterday's date
        today = datetime.now()
        endDate = (today - timedelta(days=1)).date()

        # calculate start date
        startDate = endDate - relativedelta(months=1)

        # Iterate over top-10 stocks
        for symbol in stocks:
            print(f"Collecting data for {symbol}...")
            try:
                # Request parameters: daily stock data for the past year
                request_params = StockBarsRequest(
                    symbol_or_symbols=symbol,
                    start=startDate,
                    end=endDate,
                    timeframe=TimeFrame.Hour
                )

                # Retrieve stock data
                bars = self.data_client.get_stock_bars(request_params)

                # Convert the retrieved bars to a pandas DataFrame
                df = bars.df.reset_index()
                df['symbol'] = symbol  # Add the stock symbol as a column
                stock_data.append(df)

            except Exception as e:
                print(f"Failed to collect data for {symbol}: {e}")

        # Combine all stock data into a single DataFrame
        if stock_data:
            final_df = pd.concat(stock_data, ignore_index=True)
            final_df.to_csv(self.output_path, index=False)
            print(f"Successfully collected data and saved to {self.output_path}")
            
        else:
            print("No data collected.")
        