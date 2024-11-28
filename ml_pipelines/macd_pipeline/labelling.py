import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv() 
FILE_PATH = os.getenv('FILE_PATH')
class DataLabeler:
    def __init__(self, input_path=FILE_PATH, output_path=FILE_PATH):
        self.input_path = input_path
        self.output_path = output_path

    def label_data(self, prediction_window=3, price_threshold=0.001):
        """
        Label the dataset based on MACD, Signal Line, and future price.
        Parameters:
        prediction_window: number of days into the future to analyze price changes.
        price_threshold: minimum percentage change in price to trigger buy/sell.
        """
        # load dataset
        df = pd.read_csv(self.input_path)

        # group by stock name (symbol) and apply calculations independently for each stock
        def process_group(group):
            # Calculate future price for the group
            group['future_price'] = group['close'].shift(-prediction_window)

            # Calculate price change percentage
            group['price_change'] = (group['future_price'] - group['close']) / group['close']

            # Precompute shifted MACD and Signal Line for the group
            group['macd_shifted'] = group['macd'].shift(1)
            group['signal_line_shifted'] = group['signal_line'].shift(1)

            # Apply labels based on MACD/signal line and future price
            def label_function(row):
                #if pd.isna(row['price_change']):
                if pd.isna(row['price_change']) or pd.isna(row['macd_shifted']) or pd.isna(row['signal_line_shifted']):
                    return None 
                
                # Use precomputed shifted values
                macd_above_signal = (row['macd'] > row['signal_line']) and (row['macd_shifted'] <= row['signal_line_shifted'])
                macd_below_signal = (row['macd'] < row['signal_line']) and (row['macd_shifted'] >= row['signal_line_shifted'])

                future_price_increasing = row['price_change'] > price_threshold
                future_price_decreasing = row['price_change'] < -price_threshold

                if macd_above_signal or future_price_increasing:
                    return 0  # buy
                elif macd_below_signal or future_price_decreasing:
                    return 2  # sell
                else:
                    return 1  # hold
                
              

            # Apply labeling logic
            group['label'] = group.apply(label_function, axis=1)
            
            # Make sure labels are integers
            group['label'] = group['label'].astype('Int64')

            # Drop unwanted columns
            group = group.drop(columns=['future_price', 'price_change', 'macd_shifted', 'signal_line_shifted'])

            return group

        # process each stock (symbol) separately
        df = df.groupby('symbol', group_keys=False).apply(process_group)

        # remove rows where there is no label
        df = df[df['label'].notna()]
        
        # save labeled dataset
        df.to_csv(self.output_path, index=False)
        print(f"Labeled data saved to {self.output_path}")
