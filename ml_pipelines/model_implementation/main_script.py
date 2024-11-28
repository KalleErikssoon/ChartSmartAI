
from pre_processor import Preprocessor
from model_trainer_ova import ModelTrainer
import os

def runpipeline(strategy):
    print("Starting pipeline for strategy: {strategy}...")

    model_base_path = f"ml_pipelines/model_implementation/trained_models/{strategy}"
    model_output_path = f"{model_base_path}/logistic_ova_models.pkl"

    # fetch data
    preprocessor = Preprocessor(
        api_url="http://127.0.0.1:8000/get_database/ema/",
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
    trainer.run_pipeline()


runpipeline(strategy="ema")
