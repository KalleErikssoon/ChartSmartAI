#from venv import logger
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
from ta.momentum import RSIIndicator


#Load Alpaca API keys from environment variables
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

#Initialise Alpaca client
alpaca_client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)

def fetch_stock_data(stock_symbol, start_date=None, end_date=None, timeframe=TimeFrame.Hour, min_rows=14):
    try:
        #Default: yesterday
        if end_date is None:
            end_date = (datetime.now() - timedelta(days=1)).date()
        if start_date is None:
            start_date = end_date - timedelta(days=30) #Fetch more days to ensure enough data

        print(f"Fetching stock data for {stock_symbol} from {start_date} to {end_date}")

        #Initialize data storage
        data = pd.DataFrame()

        #Continue fetching data until at least `min_rows` rows are available
        while len(data) < min_rows and end_date >= (datetime.now() - timedelta(days=30)).date():
            request_params = StockBarsRequest(
                symbol_or_symbols=stock_symbol,
                start=start_date,
                end=end_date,
                timeframe=timeframe
            )
            bars = alpaca_client.get_stock_bars(request_params)
            fetched_data = bars.df.reset_index()
            
            if not fetched_data.empty:
                data = pd.concat([data, fetched_data]).drop_duplicates().reset_index(drop=True)

            end_date -= timedelta(days=1)  #Move end_date back
            start_date = end_date - timedelta(days=30)  #Adjust start_date accordingly

        #Ensure we have enough data
        if len(data) < min_rows:
            raise ValueError(f"Not enough data fetched for RSI calculation. Required: {min_rows}, Provided: {len(data)}")

        #Sort by timestamp
        data = data.sort_values(by='timestamp').reset_index(drop=True)
        print(f"Fetched {len(data)} rows of data for {stock_symbol}")
        return data

    except Exception as e:
        raise ValueError(f"Error fetching stock data: {e}")



#Model loading function
def load_model(strategy):
    strategy = strategy.lower()
    strategy_dir = os.path.join(settings.BASE_DIR, f"stock_app/inference/models/{strategy}/")
    print(f"Looking for models in: {strategy_dir}")

    if not os.path.exists(strategy_dir):
        raise FileNotFoundError(f"No models directory found for strategy {strategy}.")
    
    #List all files in the directory
    model_files = [f for f in os.listdir(strategy_dir)]
    print(f"All files in {strategy_dir}: {model_files}")

    #Filter for .pkl files sorted in order, in order to be able to find the latest one
    pkl_files = sorted(
        [f for f in model_files if f.endswith(".pkl")],
        key=lambda x: os.path.getmtime(os.path.join(strategy_dir, x)),
        reverse=True
    )
    print(f"Filtered .pkl files: {pkl_files}")

    if not pkl_files:
        raise FileNotFoundError(f"No .pkl files found in {strategy_dir}.")
    
    #Load the most recent model
    model_path = os.path.join(strategy_dir, pkl_files[0])
    print(f"Loading model from: {model_path}")

    model = joblib.load(model_path)
    for cls, weights in model.items():
        print(f"Class {cls} weights:\n{weights}")

    return joblib.load(model_path)




#Calculate the ema for adding as a feature to the ema strategy
def calculate_ema(prices, period=10):
    """
    Calculate the Exponential Moving Average (EMA).
    """
    if len(prices) < period:
        raise ValueError(f"Not enough data to calculate EMA. Required: {period}, Provided: {len(prices)}")

    ema = np.zeros(len(prices))
    weighting_factor = 2 / (period + 1)
    sma = np.mean(prices[:period])  #Initial SMA
    ema[period - 1] = sma

    for i in range(period, len(prices)):
        ema[i] = (prices[i] * weighting_factor) + (ema[i - 1] * (1 - weighting_factor))

    ema[:period - 1] = np.nan  #NaN for values where EMA cannot be computed
    return ema


#Calculate macd and signal line for adding as features if the selected strategy is macd
def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
    """
    Calculate MACD and signal line for the given data.
    Parameters:
    - data: A pandas DataFrame with at least a 'close' column.
    - short_window: The window size for the short-term EMA.
    - long_window: The window size for the long-term EMA.
    - signal_window: The window size for the signal line EMA.
    Returns:
    - data: The DataFrame with added 'macd' and 'signal_line' columns.
    """
    if 'close' not in data.columns:
        raise KeyError("'close' column is required for MACD calculation.")

    #Calculate short-term and long-term EMAs
    short_ema = data['close'].ewm(span=short_window, adjust=False).mean()
    long_ema = data['close'].ewm(span=long_window, adjust=False).mean()

    #Calculate MACD and signal line
    data['macd'] = short_ema - long_ema
    data['signal_line'] = data['macd'].ewm(span=signal_window, adjust=False).mean()

    return data

#Calculate the RSI value in the same way as in the rsi_pipeline
def calculate_rsi(data, period=14):
    """
    Calculate RSI for the provided stock data.
    Parameters:
    - data: A DataFrame containing a 'close' column.
    - period: Period for RSI calculation (default is 14).
    Returns:
    - data: The DataFrame with an added 'rsi' column.
    """
    if 'close' not in data.columns:
        raise KeyError("'close' column is required for RSI calculation.")

    if len(data) < period:
        raise ValueError(f"Not enough data to calculate RSI. Required: {period}, Provided: {len(data)}")

    #Ensure the data is sorted by time
    data = data.sort_values(by='timestamp')

    #Calculate RSI using the `ta` library
    rsi_indicator = RSIIndicator(close=data['close'], window=period)
    data['rsi'] = rsi_indicator.rsi()  #Add the RSI column to the DataFrame

    print(data[['timestamp', 'close', 'rsi']].tail())  #Debugging: Print the last few rows of RSI
    return data




#Preprocess stock data for inference based on the selected strategy.
def preprocess_data(stock_data, strategy):
    """
    Preprocess stock data for inference based on the selected strategy.
    Parameters:
    - stock_data: The raw stock data as a DataFrame.
    - strategy: The selected strategy ('ema', 'macd', 'rsi').
    Returns:
    - processed_data: The processed data with strategy-specific features.
    """
    #Define required features for each strategy
    required_features = {
        'ema': ['open', 'high', 'low', 'close', 'volume', 'vwap', 'trade_count', 'ema'],
        'macd': ['open', 'high', 'low', 'close', 'volume', 'trade_count', 'vwap', 'macd', 'signal_line'],
        'rsi': ['open', 'high', 'low', 'close', 'volume', 'vwap', 'trade_count', 'rsi']
    }

    #Ensure the strategy is valid
    if strategy not in required_features:
        raise ValueError(f"Unknown strategy: {strategy}")

    #Base required columns for all strategies
    base_columns = ['open', 'high', 'low', 'close', 'volume', 'vwap', 'trade_count']
    if not all(col in stock_data.columns for col in base_columns):
        raise ValueError(f"Stock data is missing base columns: {base_columns}")

    # Strategy-specific calculations
    if strategy == 'ema':
        ema_period = 10  #Adjust based on the model's training
        stock_data['ema'] = calculate_ema(stock_data['close'].values, period=ema_period)

    elif strategy == 'macd':
        stock_data = calculate_macd(stock_data)

    elif strategy == 'rsi':
        rsi_period = 14 #Adjust this based on model training configuration
        stock_data = calculate_rsi(stock_data, period=rsi_period)


    #Final feature selection based on the strategy
    final_columns = required_features[strategy]
    if not all(col in stock_data.columns for col in final_columns):
        missing = [col for col in final_columns if col not in stock_data.columns]
        raise ValueError(f"Stock data is missing final columns for strategy {strategy}: {missing}")

    #Filter and drop rows with NaN values in the final feature set
    processed_data = stock_data[final_columns].dropna().iloc[-1:]
    return processed_data

#Apply manual scaling to the data using provided mean and scale values.
# def apply_scaling(data, scaler_mean, scaler_scale):
#     """
#     Parameters:
#     - data: DataFrame or numpy array to be scaled.
#     - scaler_mean: Mean values used during training.
#     - scaler_scale: Scale (standard deviation) values used during training.
#     Returns:
#     - scaled_data: Scaled version of the input data.
#     """
#     return (data - scaler_mean) / scaler_scale


#Function to make predictions
# def run_inference(processed_data, strategy):
#     try:
#         print(f"Running inference for strategy={strategy}")

#         # Load the model
#         model_dict = load_model(strategy)
#         print(f"Loaded model for strategy={strategy}: {model_dict}")

#         #Implement the OVA prediction logic
#         def sigmoid(z):
#             return 1 / (1 + np.exp(-z))

#         X = processed_data.to_numpy()
#         X = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)  #Add intercept term
        
#         probabilities = {}
#         for cls, weights in model_dict.items():
#             z = np.dot(X, weights)
#             probabilities[cls] = sigmoid(z)

#         #Choose the class with the highest probability
#         predicted_class = max(probabilities, key=lambda cls: probabilities[cls])
        
#         print(f"Predicted class: {predicted_class} with probabilities: {probabilities}")

#         #Map predicted class to action
#         action_map = {0: "Buy", 1: "Hold", 2: "Sell"}
#         predicted_action = action_map.get(predicted_class, "Unknown")
#         return predicted_action

#     except Exception as e:
#         print(f"Error during inference: {e}")
#         raise ValueError(f"Error during inference: {e}")


def run_inference(processed_data, strategy):
    try:
        print(f"Running inference for strategy={strategy}")

        # Load the model
        strategy_dir = os.path.join(settings.BASE_DIR, f"stock_app/inference/models/{strategy}/")
        print(f"Looking for models in: {strategy_dir}")
        chosen_model_file_path = os.path.join(settings.BASE_DIR, f"stock_app/chosen_model/{strategy}.txt")
        print(f"Looking for chosen strategy in: {chosen_model_file_path}")

        # # Find the latest model file
        # model_files = [f for f in os.listdir(strategy_dir) if f.endswith(".pkl") and not f.endswith("_scaler.pkl")]
        # model_files = sorted(
        #     model_files, 
        #     key=lambda x: os.path.getmtime(os.path.join(strategy_dir, x)), 
        #     reverse=True
        # )
        # if not model_files:
        #     raise FileNotFoundError(f"No model files found in {strategy_dir}")

        model_path = None

        #Check if the admin has chosen a model
        chosen_model = None
        if os.path.exists(chosen_model_file_path):
            with open(chosen_model_file_path, "r") as file:
                chosen_model = file.read().strip() #read the chosen model name
                print(f"Chosen model from {strategy}.txt: {chosen_model}")

        if chosen_model:
            model_path = os.path.join(strategy_dir, chosen_model)
            if not os.path.exists(model_path):
                print(f"Chosen model file does not exist: {model_path}")
                print("Falling back to the latest available model...")
                model_path = None  # Reset to trigger fallback logic

        if not model_path:
            # Fall back to the latest model logic
            print("No valid chosen model specified. Falling back to the latest model...")
            model_files = [f for f in os.listdir(strategy_dir) if f.endswith(".pkl") and not f.endswith("_scaler.pkl")]
            model_files = sorted(
                model_files,
                key=lambda x: os.path.getmtime(os.path.join(strategy_dir, x)),
                reverse=True
            )
            if not model_files:
                raise FileNotFoundError(f"No model files found in {strategy_dir}")

            latest_model_file = model_files[0]
            model_path = os.path.join(strategy_dir, latest_model_file)
            print(f"Loading the latest model from: {model_path}")

        # Load the model
        model_dict = joblib.load(model_path)
        print(f"Loaded model for strategy={strategy}: {model_dict}")

        # Construct the scaler file path
        scaler_path = model_path.replace(".pkl", "_scaler.pkl")
        print(f"Looking for scaler at: {scaler_path}")

        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler file not found: {scaler_path}")

        # Load the scaler
        scaler = joblib.load(scaler_path)
        print(f"Scaler loaded successfully.")

        # Apply the scaler to the processed data
        scaled_data = scaler.transform(processed_data)
        print(f"Scaled data:\n{scaled_data}")

        # Implement the OVA prediction logic
        def sigmoid(z):
            return 1 / (1 + np.exp(-z))

        X = np.concatenate([np.ones((scaled_data.shape[0], 1)), scaled_data], axis=1)  # Add intercept term

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


