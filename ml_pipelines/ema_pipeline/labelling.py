import pandas as pd
import sqlite3

class Labeler:

    def __init__(self, csv_file_path="ml_pipelines/ema_pipeline/ema_data.csv", threshold=0.01):
        self.csv_file_path = csv_file_path
        self.threshold = threshold

    def load_data(self):
        #load CSV 
        self.data = pd.read_csv(self.csv_file_path)
        print(f"Loaded data from {self.csv_file_path}")

    def create_label(self):
        #label data as 'Buy (0)' 'Sell (2)' or 'Hold (1)' based on the next days price.
        labels = []
        for i in range(len(self.data) - 1):
            current_close = self.data.loc[i, 'close']
            next_close = self.data.loc[i + 1, 'close']
            change = (next_close - current_close)

            if change >= self.threshold:
                labels.append("0")  # Buy
            elif change <= -self.threshold:
                labels.append("2")  # Sell
            else:
                labels.append("1")  # Hold
        
        labels.append(None)  # last data point has no next day price
        self.data['label'] = labels
        print("Labels created for the data.")

    # def save_to_database(self):
    #     """Save the labeled data into the SQLite database."""
    #     conn = sqlite3.connect(self.db_path)
    #     cursor = conn.cursor()

    #     # make the table 
    #     try:
    #         cursor.execute(f"""
    #             CREATE TABLE {self.table_name} (
    #                 id INTEGER PRIMARY KEY AUTOINCREMENT,
    #                 symbol TEXT,
    #                 timestamp TEXT,
    #                 open REAL,
    #                 high REAL,
    #                 low REAL,
    #                 close REAL,
    #                 volume INTEGER,
    #                 trade_count INTEGER,
    #                 vwap REAL,
    #                 label TEXT
    #             )
    #         """)
    #         print(f"Table {self.table_name} created successfully.")
    #     except sqlite3.OperationalError:
    #         print(f"Table {self.table_name} already exists.")

    #     # insert data to table
    #     for _, row in self.data.iterrows():
    #         cursor.execute(f"""
    #             INSERT INTO {self.table_name} (
    #                 symbol, timestamp, open, high, low, close, volume, trade_count, vwap, label
    #             ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    #         """, (
    #             row['symbol'],
    #             row['timestamp'],
    #             row['open'],
    #             row['high'],
    #             row['low'],
    #             row['close'],
    #             row['volume'],
    #             row['trade_count'],
    #             row['vwap'],
    #             row['label']
    #         ))
        
    #     conn.commit()
    #     conn.close()
    #     print(f"Data successfully saved to the {self.table_name} table in the database.")

    def label_data(self):
        """Execute the entire pipline load data, label it and save it to the database."""
        self.load_data()
        self.create_label()
        print("Labeled successfully.")
