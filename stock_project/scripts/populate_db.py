import os
import sys
import json
from datetime import datetime, timedelta
import django
from django.conf import settings  # Import settings for access to API keys
from importlib.metadata import version

# Add the project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'stock_project.settings')
django.setup()

from stock_app.models import StockData
from alpaca.data import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

# Connect to Alpaca API using API key and secret key
data_client = StockHistoricalDataClient(settings.ALPACA_API_KEY, settings.ALPACA_SECRET_KEY)

# Array of top-10 stocks
stocks = ["NVDA", "AAPL", "MSFT", "AMZN", "GOOG", "META", "TSLA", "BRK.B", "TSM", "AVGO"]

startDate=datetime(2024, 11, 13)
endDate=datetime.now() - timedelta(1)

for symbol in stocks:
    # Request Parameters
    request_params = StockBarsRequest(
        symbol_or_symbols=symbol,
        start=startDate,
        end=endDate,  # Updated to use yesterday's date
        timeframe=TimeFrame.Hour
    )
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


import metadata_handler

metadata_handler = metadata_handler.DataMetadata(stocks, startDate, endDate)
metadata_handler.upload_metadata()