from data_collection import DataCollector
from feature_engineering import FeatureEngineering
from labelling import DataLabeler
import requests
import os
from dotenv import load_dotenv

load_dotenv() 
FILE_PATH = os.getenv('FILE_PATH')
URL = os.getenv('URL')

def runpipeline():
    print("Starting pipeline...")

    # Step 1: Data Collection
    collector = DataCollector()
    collector.collect_data()

    #Step 2: Feature Engineering
    featureEngineering = FeatureEngineering()
    featureEngineering.calcFeaturesAndInsert()
    
    #Step 3: Data Labelling
    labeler = DataLabeler()
    labeler.label_data()

    #Step 4: Post CSV file to Django backend
    file_path = FILE_PATH

    # API endpoint
    url = URL

    # Prepare the file for upload
    with open(file_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(url, files=files)

    # Print the response from the server
    print(response.status_code)
    print(response.json())


runpipeline()
