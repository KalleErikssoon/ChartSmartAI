from data_collection import DataCollector
from labelling import Labelling
from feature_engineering import StockDataProcessor
import requests
import os
from dotenv import load_dotenv
load_dotenv()
URL = os.getenv('URL')
FILE_PATH = os.getenv('FILE_PATH')


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
    
runpipeline()
