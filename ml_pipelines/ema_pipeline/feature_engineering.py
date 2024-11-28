# Imports
import numpy as np
import os
import pandas as pd
import os
from dotenv import load_dotenv
load_dotenv()
FILE_PATH = os.getenv('EMA_FILE_PATH')

class EmaCalculator: 

    def __init__(self, file_path=FILE_PATH, period=10):
        
        self.file_path=os.path.abspath(file_path)
        self.period = period
        self.ema_data = None #Placeholder for the dataframe

        self._load_data()

    def _load_data(self):

        #Load data from the CSV file. Raise an exception if the file is not found.
        if os.path.exists(self.file_path):
            self.ema_data = pd.read_csv(self.file_path)
            print("Data loaded successfully.")
            print(self.ema_data.head())
        else:
            raise FileNotFoundError(f"File not found: {self.file_path}")


    #calculate EMA value according to "https://dayanand-shah.medium.com/exponential-moving-average-and-implementation-with-python-1890d1b880e6"
    def exponential_moving_average(self, prices, period):
        #Add array of zeroes
        ema = np.zeros(len(prices))

        #Weighting factor, multiplier which determines relative importance of recent prices vs older prices 
        weighting_factor= 2/(period + 1)
                            
        sma = np.mean(prices[:period])

        ema[period - 1] = sma

        for i in range(period, len(prices)):
            ema[i] = (prices[i] * weighting_factor) + (ema[i - 1] * (1 - weighting_factor))

        # Set initial values before the period to NaN (not calculable), since they cant be calculated
        ema[:period - 1] = np.nan
        return ema
            
    def calculate_ema(self):

        #Calculate EMA for the 'close' prices in the data and save it to the DataFrame.
        if 'close' not in self.ema_data.columns:
            raise KeyError("'close' column not found in the data.")

        prices = self.ema_data['close'].values
        ema_values = self.exponential_moving_average(prices, self.period)
        self.ema_data['ema'] = ema_values

    def save_data(self):
        #Add the ema column to the csv file
        self.ema_data.to_csv(self.file_path, index=False)

        #Print statements for testing
        print(f"Updated CSV saved with EMA values at {self.file_path}")
        print("Updated data preview:")
        print(self.ema_data[self.ema_data['symbol'] == 'GOOG'].head())
            
        #Print statements for testing and comparisons
        self.ema_data['pandas_ema'] = self.ema_data['close'].ewm(span=self.period, adjust=False).mean()
        print(self.ema_data[['ema', 'pandas_ema']].head(15))

    def run_pipeline(self):
        """
        Execute the entire pipeline:
        1. Calculate EMA
        2. Save the updated CSV file
        3. Optionally compare custom EMA with pandas EMA
        """
        print("Starting EMA calculation pipeline...")
        self.calculate_ema()
        self.save_data()
        print("Pipeline execution completed!")