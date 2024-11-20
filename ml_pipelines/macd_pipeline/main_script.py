from data_collection import DataCollector
from feature_engineering import FeatureEngineering
from labelling import DataLabeler

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

runpipeline()
