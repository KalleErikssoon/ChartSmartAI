from data_collection import DataCollector
from labelling import Labeler
from feature_engineering import StockDataProcessor
import requests

def runpipeline():
    print("Starting pipeline...")

    # Step 1: Data Collection
    collector = DataCollector()
    collector.collect_data()

    #Step 2: Feature Engineering
    rsiCalculator = StockDataProcessor()
    rsiCalculator.process()
    

    #step 3 : data labelling
    labeler= Labeler()
    labeler.process()

    #step 4: send the csv to django project via API

    file_path = "rsi_stock_data.csv"

    # API endpoint
    url = "http://127.0.0.1:8000/db_updates/"

    # Prepare the file for upload
    with open(file_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(url, files=files)

    print(response.status_code)
    print(response.json())
    
runpipeline()
