from data_collection import DataCollector
from labelling import Labeler
def runpipeline():
    print("Starting pipeline...")

    # Step 1: Data Collection
    collector = DataCollector()
    collecteddata = collector.collect_data()
    #step 2 : data labelling
    labeler= Labeler()
    labeler.label_data()
    # Continue with subsequent pipeline steps...

runpipeline()
