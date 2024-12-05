from django.test import TestCase
from datetime import datetime
from .models import MACD_Data, EMA_Data, RSI_Data

# Create mock data for testing schema
class StockMarketDataTests(TestCase):
    def setUp(self):
        # Create test data for MACD_Data
        MACD_Data.objects.create(
            symbol="AAPL",
            timestamp=datetime(2023, 11, 20, 5, 0, 0),
            open=189.89,
            high=191.905,
            low=189.88,
            close=191.45,
            volume=46543084,
            trade_count=554091,
            vwap=191.307534,
            macd=None,
            signal_line=None,
            label=1,
        )
        
        # Create test data for EMA_Data
        EMA_Data.objects.create(
            symbol="AAPL",
            timestamp=datetime(2023, 11, 21, 5, 0, 0),
            open=191.41,
            high=191.52,
            low=189.74,
            close=190.64,
            volume=38142232,
            trade_count=450941,
            vwap=190.509559,
            ema=None,
            label=1,
        )
        
        # Create test data for RSI_Data
        RSI_Data.objects.create(
            symbol="AAPL",
            timestamp=datetime(2023, 11, 22, 5, 0, 0),
            open=190.64,
            high=192.00,
            low=190.50,
            close=191.75,
            volume=32100000,
            trade_count=400000,
            vwap=191.00,
            rsi=50.0,
            label=0,
        )
    def test_macd_data_creation(self):
        macd_data = MACD_Data.objects.get(symbol="AAPL", timestamp=datetime(2023, 11, 20, 5, 0, 0))
        self.assertEqual(macd_data.close, 191.45)
        self.assertEqual(macd_data.label, 1)
    
    def test_ema_data_creation(self):
        ema_data = EMA_Data.objects.get(symbol="AAPL", timestamp=datetime(2023, 11, 21, 5, 0, 0))
        self.assertEqual(ema_data.volume, 38142232)
        self.assertEqual(ema_data.label, 1)

    def test_rsi_data_creation(self):
        rsi_data = RSI_Data.objects.get(symbol="AAPL", timestamp=datetime(2023, 11, 22, 5, 0, 0))
        self.assertEqual(rsi_data.rsi, 50.0)
        self.assertEqual(rsi_data.label, 0)

    def test_data_integrity(self):
        # Ensure the correct number of objects are created
        self.assertEqual(MACD_Data.objects.count(), 1)
        self.assertEqual(EMA_Data.objects.count(), 1)
        self.assertEqual(RSI_Data.objects.count(), 1)

    def tearDown(self):
        # Remove all test data
        MACD_Data.objects.all().delete()
        EMA_Data.objects.all().delete()
        RSI_Data.objects.all().delete()