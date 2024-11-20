from django.db import models

# Create your models here.

# Model for Stock dataset
class StockData(models.Model):
    timestamp = models.DateTimeField()  # Using DateTimeField for date and time
    symbol = models.CharField(max_length=10)  # Stock ticker symbol
    open = models.FloatField()  # Opening price
    high = models.FloatField()  # Highest price
    low = models.FloatField()  # Lowest price
    close = models.FloatField()  # Closing price
    volume = models.IntegerField()  # Volume of stocks traded
    vwap = models.FloatField()  # Volume-weighted average price
    trade_count = models.IntegerField()  # Number of trades


class MACD_Data(models.Model):
    timestamp = models.DateTimeField()  # Using DateTimeField for date and time
    symbol = models.CharField(max_length=10)  # Stock ticker symbol
    open = models.FloatField()  # Opening price
    high = models.FloatField()  # Highest price
    low = models.FloatField()  # Lowest price
    close = models.FloatField()  # Closing price
    volume = models.IntegerField()  # Volume of stocks traded
    vwap = models.FloatField()  # Volume-weighted average price
    trade_count = models.IntegerField()  # Number of trades
    macd = models.FloatField(null=True, blank=True) # MACD
    signal_line = models.FloatField(null=True, blank=True) # singal line
    macd_histogram = models.FloatField(null=True, blank=True) # MACD - signal line
    label = models.IntegerField(null=True, blank=True)  # Class Label from Future - Buy, Hold, or Sell (0, 1, 2)
