import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from datetime import datetime
#from ml_pipelines.ema_pipeline.data_collection import DataCollector  

'''
Tests:
1. Ensures "collect_data" correctly loops through all stock symbols and fetches their data.
2. Validates the logic for saving data to a file.
3. Checks for mismatches in API call count or file save parameters.
4. Verifies that mocked API data with multiple fields is handled correctly.
'''


import sys
import os

# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ml_pipelines.ema_pipeline.data_collection import DataCollector


class TestDataCollector(unittest.TestCase):

    @patch("ml_pipelines.ema_pipeline.data_collection.StockHistoricalDataClient")
    @patch("ml_pipelines.ema_pipeline.data_collection.pd.DataFrame.to_csv")

    def test_collect_data_success(self, mock_to_csv, mock_data_client):
        """
        Test the collect_data method for successful data retrieval and CSV saving.
        """
        # Mock the StockHistoricalDataClient instance and its get_stock_bars method
        mock_client_instance = mock_data_client.return_value
        mock_bars = MagicMock()  # Create a mock object for the response of get_stock_bars

        # Mock the DataFrame returned by get_stock_bars
        mock_bars.df = pd.DataFrame({
            "timestamp": [datetime(2024, 11, 20), datetime(2024, 11, 21)],  
            "open": [100, 102],  
            "high": [105, 107],  
            "low": [98, 101],  
            "close": [104, 106],  
            "volume": [1000, 1200],  
            "vwap": [102.5, 104.5],  
            "trade_count": [300, 350]  
        })
        mock_client_instance.get_stock_bars.return_value = mock_bars  # Set the mock response

        # Initialize DataCollector with mock API keys and an output file path
        collector = DataCollector(api_key="fake_api_key", secret_key="fake_secret_key", output_path="ema_test.csv")

        # Call the collect_data method to be tested
        collector.collect_data()

        # Verify that get_stock_bars was called the correct number of times
        api_call_count = mock_client_instance.get_stock_bars.call_count  # Count the number of API calls
        print(f"API called {api_call_count} times.")  # Debug: Print API call count
        
        # Assert that the API was called 10 times ( 10 stock symbol)
        self.assertEqual(mock_client_instance.get_stock_bars.call_count, 10, "Expected API call count does not match.")

        # Verify that to_csv was called correctly with the expected file name and parameters
        try:
            mock_to_csv.assert_called_once_with("ema_test.csv", index=False)  # Check if to_csv was called once with correct arguments
            print(f"File saved successfully with parameters: {mock_to_csv.call_args}.")  # Debug: Print to_csv call arguments
        except AssertionError as e:
            print(f"File save failed: {e}")  # Debug: Print error if the assertion fails
            raise  # Re-raise the exception to fail the test

if __name__ == "__main__":
    unittest.main()  # Run the test case
