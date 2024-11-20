import pandas as pd
import sqlite3

conn = sqlite3.connect('stock_project/db.sqlite3')

# query to get the table to DataFrame
query = "SELECT * FROM stock_app_stockdata"
data = pd.read_sql_query(query, conn)

def create_label(data, threshold=0.01):
    """
    Label data as 'Buy (0)' , 'Sell(2)', or 'Hold(1)' based on the next day
    Args:
        data (pd.DataFrame): Stock data with a 'close' column.
        threshold (float): Percentage change threshold for labeling.
    Returns:
        pd.DataFrame: Original data with an added 'label' column.
    """
    labels = []
    for i in range(len(data) - 1):
        current_close = data.loc[i, 'close']
        next_close = data.loc[i + 1, 'close']
        change = (next_close - current_close) 

        if change >= threshold:
            labels.append("0")  # Buy
        elif change <= -threshold:
            labels.append("2")  # Sell
        else:
            labels.append("1")  # Hold
    
    labels.append(None)  #last data point has no next day price
    data['label'] = labels
    return data

# add Labels to Data
labeled_data = create_label(data)

# database with Labels
cursor = conn.cursor()

# add a lbel column 
try:
    cursor.execute("ALTER TABLE stock_app_stockdata ADD COLUMN label TEXT")
except sqlite3.OperationalError:
    # if already exists
    pass

# updat each row in the database with the new labels
for index, row in labeled_data.iterrows():
    if row['label'] is not None:  # Skip rows with None as the label
        cursor.execute(
            "UPDATE stock_app_stockdata SET label = ? WHERE id = ?",
            (row['label'], row['id'])  # Update the 'label' column based on 'id'
        )

# commit ot database and close connection
conn.commit()
conn.close()

print("Database updated successfully with labels.")