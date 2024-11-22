
from pre_processor import Preprocessor

def runpipeline():
    print("Starting pipeline...")

    # Step 1: Fetch data
    preprocessor= Preprocessor()
    data = preprocessor.fetch_data()
    print(data)
    print("Data fetched successfully")

runpipeline()
