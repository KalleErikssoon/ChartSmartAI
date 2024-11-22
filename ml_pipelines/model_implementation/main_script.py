
from pre_processor import Preprocessor

def runpipeline():
    print("Starting pipeline...")

    # fetch data
    preprocessor = Preprocessor(api_url="http://127.0.0.1:8000/get_database/macd/")
    data = preprocessor.fetch_data()
    print("Data fetched successfully")

    # separate features and label
    feature_columns = [col for col in data.columns if col not in ['label', 'symbol']]
    X = data[feature_columns]
    y = data['label'] 

    # split data into training and testing with stratification for even distribution of different stocks in dataset
    X_train, X_test, y_train, y_test = preprocessor.split_data(X, y, stratify=data['symbol'])

    # save to csv files
    X_train.to_csv("ml_pipelines/model_implementation/X_train.csv", index=False)
    X_test.to_csv("ml_pipelines/model_implementation/X_test.csv", index=False)
    y_train.to_csv("ml_pipelines/model_implementation/y_train.csv", index=False)
    y_test.to_csv("ml_pipelines/model_implementation/y_test.csv", index=False)

    print("Data successfully split and saved to CSV files.")


runpipeline()
