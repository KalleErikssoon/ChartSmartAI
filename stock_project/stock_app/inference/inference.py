from venv import logger
from django.conf import settings
import pandas as pd
import joblib
import os
from alpaca.data import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime, timedelta
from django.http import JsonResponse
import numpy as np

# Load Alpaca API keys from environment variables
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

#Initialise Alpaca client
alpaca_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)

def fetch_stock_data(stock_symbol, start_date=None, end_date=None, timeframe=TimeFrame.Hour):
    try:
        # Default: yesterday
        if end_date is None:
            end_date = (datetime.now() - timedelta(days=1)).date()
        if start_date is None:
            start_date = end_date - timedelta(days=5)  # Default range: last 5 days

        print(f"Fetching stock data for {stock_symbol} from {start_date} to {end_date}")

        # Iteratively find the most recent available data (for cases where end_date is a weekend/holiday)
        data = pd.DataFrame()
        while data.empty and end_date >= (datetime.now() - timedelta(days=15)).date():
            print(f"No data found for {end_date}, trying previous day...")
            request_params = StockBarsRequest(
                symbol_or_symbols=stock_symbol,
                start=start_date,
                end=end_date,
                timeframe=timeframe
            )
            bars = alpaca_client.get_stock_bars(request_params)
            data = bars.df.reset_index()
            if not data.empty:
                break  # Break loop once data is fetched
            end_date -= timedelta(days=1)  # Move end_date one day back
            start_date = end_date - timedelta(days=5)  # Adjust start_date accordingly

        # Ensure data is available
        if data.empty:
            raise ValueError(f"No data fetched for {stock_symbol} within the last 15 days.")

        # Filter for the most recent day's data
        data['date'] = data['timestamp'].dt.date
        most_recent_date = data['date'].max()
        most_recent_data = data[data['date'] == most_recent_date]

        if most_recent_data.empty:
            raise ValueError(f"No data available for {stock_symbol} on the most recent available date ({most_recent_date}).")

        # Select the last available datapoint for the most recent date
        last_datapoint = most_recent_data.iloc[-1:]
        print(f"Last datapoint for {stock_symbol} on {most_recent_date}:\n", last_datapoint)
        return last_datapoint

    except Exception as e:
        raise ValueError(f"Error fetching stock data: {e}")



    

#Model loading function
def load_model(strategy):
    strategy = strategy.lower()
    strategy_dir = os.path.join(settings.BASE_DIR, f"stock_app/inference/models/{strategy}/")
    print(f"Looking for models in: {strategy_dir}")

    if not os.path.exists(strategy_dir):
        raise FileNotFoundError(f"No models directory found for strategy {strategy}.")
    
    # List all files in the directory
    model_files = [f for f in os.listdir(strategy_dir)]
    print(f"All files in {strategy_dir}: {model_files}")

    # Filter for .pkl files
    pkl_files = sorted(
        [f for f in model_files if f.endswith(".pkl")],
        key=lambda x: os.path.getmtime(os.path.join(strategy_dir, x)),
        reverse=True
    )
    print(f"Filtered .pkl files: {pkl_files}")

    if not pkl_files:
        raise FileNotFoundError(f"No .pkl files found in {strategy_dir}.")
    
    # Load the most recent model
    model_path = os.path.join(strategy_dir, pkl_files[0])
    print(f"Loading model from: {model_path}")

    return joblib.load(model_path)


def calculate_ema(prices, period=10):
    """
    Calculate the Exponential Moving Average (EMA).
    """
    if len(prices) < period:
        raise ValueError(f"Not enough data to calculate EMA. Required: {period}, Provided: {len(prices)}")

    ema = np.zeros(len(prices))
    weighting_factor = 2 / (period + 1)
    sma = np.mean(prices[:period])  # Initial SMA
    ema[period - 1] = sma

    for i in range(period, len(prices)):
        ema[i] = (prices[i] * weighting_factor) + (ema[i - 1] * (1 - weighting_factor))

    ema[:period - 1] = np.nan  # NaN for values where EMA cannot be computed
    return ema

def preprocess_data(stock_data):
    # Ensure required columns exist
    required_columns = ['open', 'high', 'low', 'close', 'volume', 'vwap', 'trade_count']
    if not all(col in stock_data.columns for col in required_columns):
        raise ValueError(f"Stock data is missing required columns: {required_columns}")

    # Calculate EMA and append it to the DataFrame
    ema_period = 1  # Adjust based on the model
    stock_data['ema'] = calculate_ema(stock_data['close'].values, period=ema_period)

    # Select the final set of features
    feature_columns = ['open', 'high', 'low', 'close', 'volume', 'vwap', 'trade_count', 'ema']
    processed_data = stock_data[feature_columns].dropna()  # Remove rows with NaN EMA values
    return processed_data



#Function to make predictions
def run_inference(processed_data, strategy):
    try:
        print(f"Running inference for strategy={strategy}")

        # Load the model
        model_dict = load_model(strategy)
        print(f"Loaded model for strategy={strategy}: {model_dict}")

        # Implement the OVA prediction logic
        def sigmoid(z):
            return 1 / (1 + np.exp(-z))

        X = processed_data.to_numpy()
        X = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)  # Add intercept term
        
        probabilities = {}
        for cls, weights in model_dict.items():
            z = np.dot(X, weights)
            probabilities[cls] = sigmoid(z)

        # Choose the class with the highest probability
        predicted_class = max(probabilities, key=lambda cls: probabilities[cls])
        
        print(f"Predicted class: {predicted_class} with probabilities: {probabilities}")

        # Map predicted class to action
        action_map = {0: "Buy", 1: "Hold", 2: "Sell"}
        predicted_action = action_map.get(predicted_class, "Unknown")
        return predicted_action

    except Exception as e:
        print(f"Error during inference: {e}")
        raise ValueError(f"Error during inference: {e}")


    

