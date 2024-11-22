import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import requests
from io import StringIO 

class Preprocessor:
    def __init__(self, api_url, test_size=0.2, random_state=None):
        self.api_url = api_url
        self.test_size = test_size
        self.random_state = random_state

    def fetch_data(self):
        # Make a GET request to the API
        response = requests.get(self.api_url)

        if response.status_code == 200:
            # Load the CSV response into a dataframe
            csv_data = StringIO(response.text)
            try:
                data = pd.read_csv(csv_data)
                if data.empty:
                    raise ValueError("No data found in the CSV file.")
                return data
            except pd.errors.EmptyDataError:
                raise ValueError("The CSV file is empty.")
            except Exception as e:
                raise ValueError(f"Failed to parse CSV data: {e}")
        else:
            # Handle errors
            raise Exception(f"Failed to fetch data: {response.status_code} - {response.text}")

    def split_data(self, X, y, stratify=None):
      # split data into training and testing sets with stratification (splitting evenly between different stocks for better generalization)
      return train_test_split(X, y, test_size=self.test_size, random_state=self.random_state, stratify=stratify)
