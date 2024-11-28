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

# Metadata preparation
metadata = {
    "name of file": os.path.basename(__file__),
    "description": "Raw data with model columns: ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'vwap', 'trade_count']",
    "origin": f"Alpaca API, python library version: {version('alpaca-py') if version('alpaca-py') else 'unknown'}",
    "date of stockmarket data": str((datetime.now() - timedelta(1)).date()),
    "date collected": str(datetime.now())
}

for symbol in stocks:
    # Request Parameters
    request_params = StockBarsRequest(
        symbol_or_symbols=symbol,
        start=datetime(2024, 11, 13),
        end=datetime.now() - timedelta(1),  # Updated to use yesterday's date
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

# Write metadata to a JSON file
metadata_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_metadata.json")
with open(metadata_file_path, "w") as metadata_file:
    json.dump(metadata, metadata_file, indent=4)

print(f"Successfully inserted data and saved metadata to {metadata_file_path}")
