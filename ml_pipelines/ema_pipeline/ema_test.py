import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from datetime import datetime
from data_collection import DataCollector  

class TestDataCollector(unittest.TestCase):

    @patch("data_collection.StockHistoricalDataClient")
    @patch("data_collection.pd.DataFrame.to_csv")
    def test_collect_data_success(self, mock_to_csv, mock_data_client):
        """
        Test the collect_data method for successful data retrieval and CSV saving.
        """
        # Mock StockHistoricalDataClient and its get_stock_bars method
        mock_client_instance = mock_data_client.return_value
        mock_bars = MagicMock()
        mock_bars.df = pd.DataFrame({
            "timestamp": [datetime(2023, 11, 20), datetime(2023, 11, 21)],
            "open": [100, 102],
            "high": [105, 107],
            "low": [98, 101],
            "close": [104, 106],
            "volume": [1000, 1200],
        })
        mock_client_instance.get_stock_bars.return_value = mock_bars

        # Initialize DataCollector with mock API keys and output path
        collector = DataCollector(api_key="fake_api_key", secret_key="fake_secret_key", output_path="ema_test.csv")

        # Call the method to test
        collector.collect_data()

        # Check if get_stock_bars was called for each stock
        self.assertEqual(mock_client_instance.get_stock_bars.call_count, 10)

        # Check if to_csv was called to save the final DataFrame
        mock_to_csv.assert_called_once_with("ema_test.csv", index=False)

if __name__ == "__main__":
    unittest.main()
