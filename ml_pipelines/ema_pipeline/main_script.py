from data_collection import DataCollector
from labelling import Labeler
from feature_engineering import EmaCalculator
import requests
import os
from dotenv import load_dotenv
load_dotenv()
FILE_PATH = os.getenv('EMA_FILE_PATH')
URL = os.getenv('EMA_URL')

def runpipeline():
    print("Starting pipeline...")

    # Step 1: Data Collection
    collector = DataCollector()
    collecteddata = collector.collect_data()

    #Step 2 : ema column 
    #Call the feature_engineering.py
    emaCalculator = EmaCalculator()
    emaCalculator.run_pipeline()
    

    #step 3 : data labelling
    labeler= Labeler()
    labeler.label_data()

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
    STRATEGY = "EMA"

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

    # clean up
    os.remove(f"{STRATEGY.lower()}_data.csv")
    
runpipeline()
