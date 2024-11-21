from data_collection import DataCollector
from feature_engineering import FeatureEngineering
from labelling import DataLabeler
import requests

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
    # File path to the CSV file
    file_path = "ml_pipelines/macd_pipeline/macd_data.csv"

    # API endpoint of your Django project
    url = "http://127.0.0.1:8000/db_updates/"  # Replace with your actual endpoint

    # Prepare the file for upload
    with open(file_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(url, files=files)

    # Print the response from the server
    print(response.status_code)
    print(response.json())


runpipeline()
