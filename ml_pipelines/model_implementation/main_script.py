
import joblib
from pre_processor import Preprocessor
from model_trainer_ova import ModelTrainer
import sys
import os
from dotenv import load_dotenv
from datetime import datetime 
import requests

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
    print("Starting pipeline for strategy: {strategy}...")

    model_base_path = f"{FILE_PATH}/trained_models/{strategy}"
    model_output_path = f"{model_base_path}/ova_{strategy}_{current_datetime}.pkl"
    scaler_output_path = model_output_path.replace(".pkl", "_scaler.pkl")


    # fetch data
    preprocessor = Preprocessor(
        api_url=f"{URL}/get_database/{strategy}",
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

    joblib.dump(preprocessor.scaler, scaler_output_path)
    print(f"Scaler saved to {scaler_output_path}")

    # Send both the model and scaler files to the Django project
    send_model_and_scaler(model_output_path, scaler_output_path, URL, {
        "strategy": strategy.lower(),
        "timestamp": current_datetime
    })


    # this is to post the picke file to the django project
    # we can move this to another place later on

    #current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    # payload = {
    #     "strategy": strategy.lower(),
    #     "timestamp": current_datetime
    # }

    # with open(model_output_path, 'rb') as f:
    #     response = requests.post(
    #         f"{URL}/upload_model/", 
    #         files={'file': f},
    #         data=payload
    #     )
    # if response.status_code == 201:
    #     print("Pickle model file is posted successfully")
    # else:
    #     print(f"Failed to upload file: {response.content}")

    

    

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

