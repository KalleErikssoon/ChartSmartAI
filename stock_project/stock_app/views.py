from django.shortcuts import render

# Create your views here.
# Home view
def home(request):
    return render(request, "stock_project/home.html", {})



# Data validation view
import json
from datetime import datetime, timedelta
from django.utils import timezone
from django.http import JsonResponse
from stock_app.models import StockData

def validate_stock_data(request):
    try:
        # Load the metadata from the JSON file
        with open('./metadata/metadata_StockData.json', 'r') as file:
            metadata = json.load(file)
        
        # Extract the stock market date from the metadata
        stockmarket_data_date = datetime.strptime(metadata['date of stockmarket data'], "%Y-%m-%d")

        # Convert the date to a timezone-aware datetime object
        stockmarket_data_date = timezone.make_aware(stockmarket_data_date)

        results = []

        # Test 1: Ensure no timestamps exceed the metadata date
        threshold_date = stockmarket_data_date + timedelta(days=1)
        invalid_entries = StockData.objects.filter(timestamp__gt=threshold_date)
        if invalid_entries.count() > 0:
            results.append({
                "test": "timestamps_not_exceeding_metadata_date",
                "status": "failed",
                "message": f"Found {invalid_entries.count()} entries with timestamp exceeding the metadata date"
            })
        else:
            results.append({
                "test": "timestamps_not_exceeding_metadata_date",
                "status": "passed"
            })

        # Test 2: Validate the schema of the StockData model
        expected_schema = ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'vwap', 'trade_count']
        actual_schema = [field.name for field in StockData._meta.get_fields() if field.name != 'id']
        if actual_schema != expected_schema:
            results.append({
                "test": "schema_validation",
                "status": "failed",
                "message": "Actual schema does not match expected schema"
            })
        else:
            results.append({
                "test": "schema_validation",
                "status": "passed"
            })

        # Test 3: Ensure each stock symbol has at least one entry
        stock_symbols = metadata['stocks']
        for symbol in stock_symbols:
            stock_count = StockData.objects.filter(symbol=symbol).count()
            if stock_count < 1:
                results.append({
                    "test": "stocks_have_entries",
                    "status": "failed",
                    "message": f"No entries found for stock symbol {symbol}"
                })
            else:
                results.append({
                    "test": "stocks_have_entries",
                    "status": "passed",
                    "symbol": symbol
                })

        return JsonResponse({"results": results, "status": "completed"})

    except Exception as e:
        return JsonResponse({"status": "error", "message": str(e)}, status=500)
