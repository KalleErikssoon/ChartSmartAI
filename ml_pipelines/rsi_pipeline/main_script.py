from data_collection import DataCollector
from labelling import Labeler
from feature_engineering import RSIIndicator
import requests

def runpipeline():
    print("Starting pipeline...")

    # Step 1: Data Collection
    collector = DataCollector()
    collector.collect_data()

    #Step 2: Feature Engineering
    rsiCalculator = RSIIndicator
    rsiCalculator.run_pipeline()
    

    #step 3 : data labelling
    labeler= Labeler()
    labeler.label_data()

    #step 4: send the csv to django project via API

    file_path = "ml_pipelines/rsi_pipeline/rsi_stock_data.csv"

    # API endpoint
    url = "http://127.0.0.1:8000/db_updates/ema/"

    # Prepare the file for upload
    with open(file_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(url, files=files)

    print(response.status_code)
    print(response.json())
    
runpipeline()
