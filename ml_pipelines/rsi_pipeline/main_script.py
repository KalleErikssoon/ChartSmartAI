from data_collection import DataCollector
from labelling import Labelling
from feature_engineering import StockDataProcessor
import requests
import os
from dotenv import load_dotenv
load_dotenv()
URL = os.getenv('RSI_URL')
FILE_PATH = os.getenv('RSI_FILE_PATH')


def runpipeline():
    print("Starting pipeline...")

    # Step 1: Data Collection
    collector = DataCollector()
    collector.collect_data()

    #Step 2: Feature Engineering
    rsiCalculator = StockDataProcessor()
    rsiCalculator.process()
    

    #step 3 : data labelling
    labeler= Labelling()
    labeler.process()

    #step 4: send the csv to django project via API
    file_path = FILE_PATH

    # API endpoint
    url = URL

    # Prepare the file for upload
    with open(file_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(url, files=files)

    print(response.status_code)
    print(response.json())

    ## Step 5: Metadata
    import metadata_handler
    import pandas as pd
    # Set the strategy
    STRATEGY = "RSI"

    fileName = FILE_PATH
    description = f"{STRATEGY} stock data for the top-10 stocks"
    model = f"STOCK_APP_{STRATEGY}_DATA"

    # Read the columns from the csv file data frame
    df = pd.read_csv(fileName)
    schema = list(df.columns)
    
    # Read off the timestamps
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    startDate = df['timestamp'].min()
    endDate = df['timestamp'].max()
    
    # Get unique stock symbols
    stocks = df['symbol'].unique().tolist()
    metadata_handler = metadata_handler.DataMetadata(fileName, description, stocks, model, schema, startDate, endDate)
    metadata_handler.upload_metadata()

runpipeline()
