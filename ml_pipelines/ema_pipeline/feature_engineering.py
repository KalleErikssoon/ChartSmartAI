import numpy as np
import os
import pandas as pd

#Import the csv file
file_path = os.path.abspath("ml_pipelines/ema_pipeline/ema_data.csv")

# Check to see if the ema_data.csv file exists 
if os.path.exists(file_path):
    #Read the data from the file
    ema_data = pd.read_csv(file_path)
    print(ema_data.head())
#If the file is not found, notify    
else:
    raise FileNotFoundError(f"File not found: {file_path}")


#calculate EMA value
def exponential_moving_average(prices, period):
    ema = np.zeros(len(prices))

    weighting_factor= 2/(period + 1)
                         
    sma = np.mean(prices[:period])

    ema[period - 1] = sma

    for i in range(period, len(prices)):
        ema[i] = (prices[i] * weighting_factor) + (ema[i - 1] * (1 - weighting_factor))

    # Set initial values before the period to NaN (not calculable)
    ema[:period - 1] = np.nan
    return ema
    

period = 10
ema_values = exponential_moving_average(ema_data['close'].values, period)

ema_data['ema'] = ema_values

ema_data.to_csv(file_path, index=False)

print(f"Updated CSV saved with EMA values at {file_path}")
print("Updated data preview:")
print(ema_data[ema_data['symbol'] == 'GOOG'].head())
    


ema_data['pandas_ema'] = ema_data['close'].ewm(span=period, adjust=False).mean()
print(ema_data[['ema', 'pandas_ema']].head(15))