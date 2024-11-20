from data_collection import DataCollector
from labelling import DataLabeler

def runpipeline():
    print("Starting pipeline...")

    # Step 1: Data Collection
    collector = DataCollector()
    collector.collect_data()

    #Step 2: Data Labeling
    labeler = DataLabeler()
    labeler.calcLabelsAndInsert()
    

runpipeline()
