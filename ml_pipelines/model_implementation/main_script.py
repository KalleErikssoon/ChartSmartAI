# Author: Karl Eriksson, Malte Bengtsson, Nasit Vurgun
import joblib
from pre_processor import Preprocessor
from model_trainer_ova import ModelTrainer
import sys
import os
from dotenv import load_dotenv
from datetime import datetime 
import requests
from metadata_handler import DataMetadata
import pandas as pd

load_dotenv() 
FILE_PATH = os.getenv('FILE_PATH')
URL = os.getenv('URL')


def send_model_and_scaler(model_output_path, scaler_output_path, URL, payload):
    # Open the model and scaler files
        with open(model_output_path, 'rb') as model_file, open(scaler_output_path, 'rb') as scaler_file:
            # Send both files in the same POST request
            response = requests.post(
                f"{URL}/upload_model/", 
                files={
                    'model_file': model_file,
                    'scaler_file': scaler_file,
                },
                data=payload
            )
        if response.status_code == 201:
            print("Model and scaler files posted successfully.")
        else:
            print(f"Failed to upload files: {response.content}")
        return response

def runpipeline(strategy, current_datetime):
    print(f"Starting pipeline for strategy: {strategy}...")  # Corrected f-string
    
    # Constructing the model base path
    model_base_path = os.path.join(FILE_PATH, "trained_models", strategy)
    
    # Ensuring the directory exists
    os.makedirs(model_base_path, exist_ok=True)
    
    # Constructing the full model output path
    model_output_path = os.path.join(
        model_base_path, f"ova_{strategy}_{current_datetime}.pkl"
    )
    
    # Now, `model_base_path` and `model_output_path` are properly set up

    # fetch data
    preprocessor = Preprocessor(
        api_url=f"{URL}/get_database/{strategy}/",
        apply_smote=True,
        apply_scaling=True)
    data = preprocessor.fetch_data()
    print("Data fetched successfully")

    # separate features and label
    feature_columns = [col for col in data.columns if col not in ['label', 'symbol', 'timestamp']]
    X = data[feature_columns]
    y = data['label'] 

    # split data into training and testing with stratification for even distribution of different stocks in dataset
    X_train, X_test, y_train, y_test = preprocessor.split_and_preprocess_data(X, y, stratify=data['symbol'])

    # save to csv files
    X_train.to_csv(f"{model_base_path}/X_train.csv", index=False)
    X_test.to_csv(f"{model_base_path}/X_test.csv", index=False)
    y_train.to_csv(f"{model_base_path}/y_train.csv", index=False)
    y_test.to_csv(f"{model_base_path}/y_test.csv", index=False)

    print(f"Data successfully split and saved for {strategy}.")

    #Instantiate and run the modeltrainer class
    trainer = ModelTrainer(
        train_data_path=model_base_path, 
        test_data_path=model_base_path, 
        model_output_path=model_output_path
    )
    trainer.scaler = preprocessor.scaler
    trainer.run_pipeline()

    # Change the filepath of the scaler file
    scaler_filepath = f"ova_{strategy}_{current_datetime}_scaler.pkl"
    os.rename("scaler_params.pkl", scaler_filepath)

    # this is to post the picke file to the django project
    # we can move this to another place later on

    with open(model_output_path, 'rb') as f:
        response = requests.post(f"{URL}/upload_model/", files={'file': f})

    if response.status_code == 201:
        print("Pickle model file is posted successfully")
        
        # post the scaler file to the django project
        with open(scaler_filepath, 'rb') as f:
            response = requests.post(f"{URL}/upload_model/", files={'file': f})

        if response.status_code == 201:
            print("Scaler pickle file is posted successfully")
        else:
            print(f"Failed to upload file: {response.content}")

        # post the metadata file to the django project
        model_filepath = f"ova_{strategy}_{current_datetime}.pkl"
        description = f"{strategy} one-vs-all classifier model" 
        database_class = f"STOCK_APP_{strategy.upper()}_DATA"
        metadata_file_path = f"ova_{strategy}_{current_datetime}_metadata.json"
        performance_json_filepath = f"{model_base_path}/ova_{strategy}_{current_datetime}_performance.json"
        metadata_handler = DataMetadata(strategy.upper(), model_filepath, scaler_filepath, description, database_class, current_datetime, metadata_file_path, performance_json_filepath)
        metadata_handler.upload_metadata()

        # Remove the scaler file
        os.remove(scaler_filepath)

        # Remove the model file
        os.remove(model_output_path)

        # Remove trained models directory even if it is not empty
        #os.rmdir(model_base_path)
        
    else:
        print(f"Failed to upload file: {response.content}")

    
#Run the mainscript pipeline. Currently hardcoded strategy, this will be dynamically received from the django project via http (from the admin page)
#I.e admin sends a http message via an url endpoint that contains either "ema", "macd" or "rsi" instead of hardcoding it here
#runpipeline(strategy="ema")

#This is the dynamic way we will run the model pipeline. This way we can dynamically run the pipeline via sent strategy arguments ema, macd or rsi
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main_script.py <strategy>")
        sys.exit(1)
    strategy = sys.argv[1]
    current_datetime = datetime.now().strftime("%Y%m%d_%H-%M--%S")
    runpipeline(strategy, current_datetime)
