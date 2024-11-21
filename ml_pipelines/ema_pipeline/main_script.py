from data_collection import DataCollector
from labelling import Labeler
from feature_engineering import EmaCalculator
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
    # Continue with subsequent pipeline steps...

runpipeline()
