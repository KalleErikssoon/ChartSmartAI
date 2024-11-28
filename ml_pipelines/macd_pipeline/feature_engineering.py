import pandas as pd  

import os
from dotenv import load_dotenv

load_dotenv() 
FILE_PATH = os.getenv('FILE_PATH')

class FeatureEngineering:
    def __init__(self, input_path=FILE_PATH, output_path=FILE_PATH):
        
        self.output_path = output_path
        self.input_path = input_path
    
    def calcFeaturesAndInsert(self):
        short_window = 12
        long_window = 26
        signal_window = 9

        df =  pd.read_csv(FILE_PATH)
        macd_results = []
        symbols = df['symbol'].unique()

        for symbol in symbols:

            symbol_data = df[df['symbol'] == symbol].copy()
            symbol_data = symbol_data.sort_values('timestamp')
            
            # calculate EMA
            shortEMA = symbol_data['close'].ewm(span=short_window, adjust=False).mean() 
            longEMA = symbol_data['close'].ewm(span=long_window, adjust=False).mean() 

            # Calculate MACD
            symbol_data['macd'] = shortEMA - longEMA

            #Calculate signal line
            symbol_data['signal_line'] = symbol_data['macd'].ewm(span=signal_window, adjust=False).mean()

            symbol_data = symbol_data.iloc[26:]
            #Append to dataframe
            macd_results.append(symbol_data)

        final_df = pd.concat(macd_results, ignore_index=True)
        final_df.to_csv(self.output_path, index=False)
        print(f"MACD calculation complete. Data saved to {self.output_path}")
    