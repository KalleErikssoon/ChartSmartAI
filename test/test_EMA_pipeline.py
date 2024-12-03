import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from datetime import datetime
import tempfile
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
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_path = os.path.join(self.temp_dir.name, "test.csv")

    def tearDown(self):
        self.temp_dir.cleanup()

    @patch("ml_pipelines.ema_pipeline.data_collection.StockHistoricalDataClient")
    def test_initialization_with_params(self, mock_data_client):
        collector = DataCollector(api_key="test_key", secret_key="test_secret", output_path=self.output_path)
        self.assertEqual(collector.api_key, "test_key")
        self.assertEqual(collector.secret_key, "test_secret")
        self.assertEqual(collector.output_path, self.output_path)

    

    def test_initialization_missing_keys(self):
        with self.assertRaises(ValueError):
            DataCollector(output_path=self.output_path)

    

    @patch("ml_pipelines.ema_pipeline.data_collection.StockHistoricalDataClient")
    @patch("ml_pipelines.ema_pipeline.data_collection.pd.DataFrame.to_csv")
    def test_collect_data_success(self, mock_to_csv, mock_data_client):
        mock_client_instance = mock_data_client.return_value
        mock_bars = MagicMock()
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
        mock_client_instance.get_stock_bars.return_value = mock_bars

        collector = DataCollector(api_key="fake_api_key", secret_key="fake_secret_key", output_path=self.output_path)
        collector.collect_data()

        self.assertEqual(mock_client_instance.get_stock_bars.call_count, 10)
        mock_to_csv.assert_called_once_with(self.output_path, index=False)

    @patch("ml_pipelines.ema_pipeline.data_collection.StockHistoricalDataClient")
    def test_data_processing(self, mock_data_client):
        mock_client_instance = mock_data_client.return_value
        mock_bars = MagicMock()
        mock_bars.df = pd.DataFrame({
            "timestamp": [datetime(2024, 11, 20)],
            "open": [100],
            "close": [104]
        })
        mock_client_instance.get_stock_bars.return_value = mock_bars

        collector = DataCollector(api_key="test_key", secret_key="test_secret", output_path=self.output_path)
        collector.collect_data()

        saved_df = pd.read_csv(self.output_path)
        self.assertIn("symbol", saved_df.columns)
        self.assertEqual(saved_df["symbol"].iloc[0], "NVDA")

   


if __name__ == "__main__":
    unittest.main()  # Run the test case
