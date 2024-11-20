# 0. calc EMA
# 1. calc macd & insert
# 2. calc signal line & insert
# 3. calc histogram & insert
# 4. insert new columns into csv file

# Additional imports
import pandas as pd  # To handle data storage


class FeatureEngineering:
    def __init__(self, input_path="ml_pipelines/macd_pipeline/macd_data.csv", output_path="ml_pipelines/macd_pipeline/macd_data.csv"):
        
        self.output_path = output_path
        self.input_path = input_path
    
    def calcFeaturesAndInsert(self):
        short_window = 12
        long_window = 26
        signal_window = 9

        df =  pd.read_csv("ml_pipelines/macd_pipeline/macd_data.csv")
        macd_results = []
        symbols = df['symbol'].unique()

        for symbol in symbols:

            symbol_data = df[df['symbol'] == symbol].copy()
            symbol_data = symbol_data.sort_values('timestamp')
            
            # calculate EMA
            shortEMA = symbol_data['close'].ewm(span=short_window, adjust=False).mean() 
            longEMA = symbol_data['close'].ewm(span=long_window, adjust=False).mean() 

            # Calculate MACD
            symbol_data['MACD'] = shortEMA - longEMA

            #Calculate signal line
            symbol_data['SignalLine'] = symbol_data['MACD'].ewm(span=signal_window, adjust=False).mean()

            symbol_data = symbol_data.iloc[26:]
            #Append to dataframe
            macd_results.append(symbol_data)

        final_df = pd.concat(macd_results, ignore_index=True)
        final_df.to_csv(self.output_path, index=False)
        print(f"MACD calculation complete. Data saved to {self.output_path}")
    