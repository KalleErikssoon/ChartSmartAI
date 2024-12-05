import pandas as pd
import joblib
import os
from alpaca.data import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime, timedelta
from django.http import JsonResponse

# Load Alpaca API keys from environment variables
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

#Initialise Alpaca client
alpaca_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)

def fetch_stock_data(stock_symbol, start_date, end_date, timeframe=TimeFrame.Hour):
    try:
        request_params = StockBarsRequest(
            symbol_or_symbols=stock_symbol,
            start=start_date,
            end=end_date,
            timeframe=timeframe
        )
        bars = alpaca_client.get_stock_bars(request_params)
        return bars.df.reset_index()
    except Exception as e:
        raise ValueError(f"Error fetching stock data: {e}")
    

#Model loading function
def load_model(strategy):
    

