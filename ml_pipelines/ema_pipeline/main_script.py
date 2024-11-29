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
    
runpipeline()
