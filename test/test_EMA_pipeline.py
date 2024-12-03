import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from datetime import datetime
import tempfile
#from ml_pipelines.ema_pipeline.data_collection import DataCollector  




import sys
import os

# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ml_pipelines.ema_pipeline.data_collection import DataCollector
from ml_pipelines.ema_pipeline.feature_engineering import EmaCalculator
from ml_pipelines.ema_pipeline.labelling import Labeler

# Mock test data colloection
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

# Mock test feature engineering
class TestEmaCalculator(unittest.TestCase):
    def setUp(self):
        # Create temporary files and directories
        self.temp_dir = tempfile.TemporaryDirectory()
        self.file_path = os.path.join(self.temp_dir.name, "test.csv")
        self.sample_data = pd.DataFrame({
            'timestamp': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05'],
            'symbol': ["NVDA", "AAPL", "MSFT", "AMZN", "GOOG"],
            'close': [500, 520, 510, 530, 550]
        })
        self.sample_data.to_csv(self.file_path, index=False)

    def test_initialization(self):
        calculator = EmaCalculator(file_path=self.file_path, period=10)
        self.assertEqual(calculator.file_path, os.path.abspath(self.file_path))
        self.assertEqual(calculator.period, 10)
    
    def test_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            EmaCalculator(file_path="non_existent_file.csv")

    # test if load the data correctly
    def test_load_data(self):
        calculator = EmaCalculator(file_path=self.file_path)
        self.assertIsInstance(calculator.ema_data, pd.DataFrame)
        self.assertListEqual(list(calculator.ema_data.columns), ['timestamp', 'symbol', 'close'])

# test caluclate ema function if add ema clounm to the data
    def test_calculate_ema(self):
        calculator = EmaCalculator(file_path=self.file_path, period=3)
        calculator.calculate_ema()
        self.assertIn('ema', calculator.ema_data.columns)

        #  check if  the EMA value is similar to the result calculated by Pandas
        pandas_ema = calculator.ema_data['close'].ewm(span=3, adjust=False).mean()
        self.assertTrue(
            pd.Series(calculator.ema_data['ema'][2:]).round(2).equals(pandas_ema[2:].round(2))
        )
    
    # test if save the data correctly
    def test_save_data(self):
        calculator = EmaCalculator(file_path=self.file_path, period=3)
        calculator.calculate_ema()
        calculator.save_data()

        # Reload the saved file and validate the contents
        saved_data = pd.read_csv(self.file_path)
        self.assertIn('ema', saved_data.columns)

# Mock test lableling
class TestLabeler(unittest.TestCase):
    def setUp(self):
        # Create temporary directories and CSV files
        self.temp_dir = tempfile.TemporaryDirectory()
        self.file_path = os.path.join(self.temp_dir.name, "test.csv")
        self.sample_data = pd.DataFrame({
            'timestamp': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05'],
            'close': [398, 342, 499, 500, 488],
            'ema': [1490, 1510, 1505, 1525, 1540]
        })
        self.sample_data.to_csv(self.file_path, index=False)

    def tearDown(self):
        # Clean temporary directories and CSV files
        self.temp_dir.cleanup()

    def test_initialization(self):
        labeler = Labeler(csv_file_path=self.file_path, threshold=0.02, prediction_window=2)
        self.assertEqual(labeler.csv_file_path, self.file_path)
        self.assertEqual(labeler.threshold, 0.02)
        self.assertEqual(labeler.prediction_window, 2)

    def test_load_data(self):
        labeler = Labeler(csv_file_path=self.file_path)
        labeler.load_data()
        self.assertIsInstance(labeler.data, pd.DataFrame)
        self.assertListEqual(list(labeler.data.columns), ['timestamp', 'close', 'ema'])

    def test_create_label(self):
        labeler = Labeler(csv_file_path=self.file_path, threshold=0.01, prediction_window=1)
        labeler.load_data()
        labeler.create_label()

        # Check that the label column is generated
        self.assertIn('label', labeler.data.columns)

        # validate part of the tag logic
        # Example: Future price increases, EMA increases, current price above EMA -> should be "2" (sale)
        self.assertEqual(labeler.data['label'].iloc[0], "2")
        
    def test_data_cleaning(self):
        labeler = Labeler(csv_file_path=self.file_path, threshold=0.01, prediction_window=1)
        labeler.load_data()
        labeler.create_label()

        initial_length = len(labeler.data)

    # simulate missing values
        labeler.data.loc[0, 'ema'] = None
        labeler.data.loc[1, 'label'] = None
       
        
        print("Before cleaning:")
        print(labeler.data)

        print("Original data:")
        print(labeler.data)
        print(labeler.data[labeler.data.isna().any(axis=1)])

    # Calculate the actual number of rows that should be removed
        expected_removed_rows = labeler.data[['label', 'ema']].isna().any(axis=1).sum()


        labeler.data_cleaning()
        cleaned_length = len(labeler.data)

    

        print("After cleaning:")
        print(labeler.data)

    
        self.assertEqual(cleaned_length, initial_length - expected_removed_rows)


if __name__ == "__main__":
    unittest.main()  # Run the test case
