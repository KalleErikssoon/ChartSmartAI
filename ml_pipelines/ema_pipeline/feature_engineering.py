# Imports
import numpy as np
import os
import pandas as pd

#Access the csv file via absolute path
file_path = os.path.abspath("ml_pipelines/ema_pipeline/ema_data.csv")

# Check to see if the ema_data.csv file exists 
if os.path.exists(file_path):
    #Read the data from the file
    ema_data = pd.read_csv(file_path)
    print(ema_data.head())
#If the file is not found, notify    
else:
    raise FileNotFoundError(f"File not found: {file_path}")


#calculate EMA value according to "https://dayanand-shah.medium.com/exponential-moving-average-and-implementation-with-python-1890d1b880e6"
def exponential_moving_average(prices, period):
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
    
# Amount of datapoints, for example 10 days (or hours, depending)
period = 10

ema_values = exponential_moving_average(ema_data['close'].values, period)

#Create ema column with the ema values
ema_data['ema'] = ema_values
#Add the ema column to the csv file
ema_data.to_csv(file_path, index=False)

#Print statements for testing
print(f"Updated CSV saved with EMA values at {file_path}")
print("Updated data preview:")
print(ema_data[ema_data['symbol'] == 'GOOG'].head())
    
#Print statements for testing and comparisons
ema_data['pandas_ema'] = ema_data['close'].ewm(span=period, adjust=False).mean()
print(ema_data[['ema', 'pandas_ema']].head(15))