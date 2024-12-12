import pandas as pd
import os
from dotenv import load_dotenv
load_dotenv()
FILE_PATH = os.getenv('EMA_FILE_PATH')
class Labeler:

    def __init__(self, csv_file_path=FILE_PATH, threshold=0.001, prediction_window=1):
        self.csv_file_path = csv_file_path
        self.threshold = threshold
        self.prediction_window = prediction_window  # this is number of days to predict

    #Load the data from the csv file path from the .env file
    def load_data(self):
        self.data = pd.read_csv(self.csv_file_path)
        print(f"Loaded data from {self.csv_file_path}")

    #Calculate what the label should be by taking into consideration both the change in ema value and price change over time
    def label_function(self, row):
        if pd.isna(row['price_change']):
            return None  # if there are no future price then return none

        price_above_ema = row['close'] > row['ema']
        future_price_increasing = row['price_change'] > self.threshold
        future_price_decreasing = row['price_change'] < -self.threshold
        ema_increasing = row['ema_change'] > 0
        ema_decreasing = row['ema_change'] < 0

        if (price_above_ema or ema_increasing) and future_price_increasing:
            return "0"  #buy
        elif (not price_above_ema or ema_decreasing) and future_price_decreasing:
            return "2"  #sell
        else:
            return "1"  #hold

    #Create the actual label column and call the label_function to calculate labels for each datapoint
    def create_label(self):
        if 'ema' not in self.data.columns:
            raise ValueError("The 'ema' column is missing.")

        self.data['future_price'] = self.data['close'].shift(-self.prediction_window)

        #price change percentage
        self.data['price_change'] = (self.data['future_price'] - self.data['close']) / self.data['close']

        #change in ema value over time
        self.data['ema_change'] =  (self.data['ema']) - self.data['ema'].shift(1)

        # apply the labels to rows
        self.data['label'] = self.data.apply(self.label_function, axis=1)
        print("labels created for the data.")

        #drop columns that are not needed
        self.data.drop(['future_price', 'price_change', 'ema_change'], axis=1, inplace=True)

    #Remove rows (datapoints) with missing label or ema value
    def data_cleaning(self):
        self.data.dropna(subset=['label', 'ema'], inplace=True)
        print("Removed values with missing label or EMA value.")

    def save_to_csv(self):
        self.data.to_csv(self.csv_file_path, index=False)
        print(f"data successfully saved to {self.csv_file_path}.")

    def label_data(self):
        """Load data, label it and save it to csv."""
        self.load_data()
        self.create_label()
        self.data_cleaning()
        self.save_to_csv()
        print("Labeled successfully.")
