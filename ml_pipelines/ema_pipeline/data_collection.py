# Django imports
import os
import sys
import django
from django.conf import settings # Import settings for access to api keys


# Add the project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'stock_project.settings')
django.setup()

from stock_project.stock_app.models import StockData

# Alpaca API imports
from alpaca.data import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime # to handle start and end time

# Connect to Alpaca API using API key and secret key
# See 'settings.py' for instructions on how to add you own Alpaca API keys
data_client = StockHistoricalDataClient(settings.ALPACA_API_KEY, settings.ALPACA_SECRET_KEY)

# Array of top-10 stocks
stocks = ["NVDA", "AAPL", "MSFT", "AMZN", "GOOG", "META", "TSLA", "BRK.B", "TSM", "AVGO"]

for symbol in stocks:
    # Request Parameters
    request_params = StockBarsRequest(
        symbol_or_symbols=symbol, 
        start=datetime(2024, 9, 13),
        end=datetime(2024, 11, 13),
        timeframe=TimeFrame.Hour
    )
    # Send an API request to retrieve "bar" data for the specified stock, using the given request parameters.
    # Each "bar" represents one time interval (in this case, daily) and includes columns like open, high, low, close, volume, and more.
    # The retrieved bars are stored in the 'bars' array
    bars = data_client.get_stock_bars(request_params)

    # Insert retrieved data using Django ORM
    for bar in bars[symbol]:
        StockData.objects.create(
            timestamp=bar.timestamp,
            symbol=bar.symbol,
            open=bar.open,
            high=bar.high,
            low=bar.low,
            close=bar.close,
            volume=bar.volume,
            vwap=bar.vwap,
            trade_count=bar.trade_count
        )

print("Successfully inserted data")