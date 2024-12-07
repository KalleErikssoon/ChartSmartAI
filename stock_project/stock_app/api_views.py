from datetime import datetime, timedelta
import os
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import MultiPartParser
import pandas as pd
import json
from stock_app.models import MACD_Data 
from stock_app.models import EMA_Data
from stock_app.models import RSI_Data
import os
from django.conf import settings
from stock_app.inference.inference import fetch_stock_data, load_model, run_inference, preprocess_data

@csrf_exempt
@api_view(['POST'])
@parser_classes([MultiPartParser])
def upload_csv(request):
    if 'file' not in request.FILES:
        return JsonResponse({'error': 'No file uploaded'}, status=400)

    try:
        # Read the uploaded file
        csv_file = request.FILES['file']
        df = pd.read_csv(csv_file)

        # Insert each row into the database
        for _, row in df.iterrows():
            MACD_Data.objects.create(
                symbol=row['symbol'],
                timestamp=row['timestamp'],
                open=row['open'],
                high=row['high'],
                low=row['low'],
                close=row['close'],
                volume=row['volume'],
                trade_count=row['trade_count'],
                vwap=row['vwap'],
                macd=row['macd'],
                signal_line=row['signal_line'],
                label=row['label']
            )
        return JsonResponse({'message': 'Data successfully uploaded'}, status=201)

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
@api_view(['POST'])
@parser_classes([MultiPartParser])
def upload_ema(request):
    if 'file' not in request.FILES:
        return JsonResponse({'error': 'No file uploaded'}, status=400)

    try:
        # Read the uploaded file
        csv_file = request.FILES['file']
        df = pd.read_csv(csv_file)

        # Insert each row into the database
        for _, row in df.iterrows():
            EMA_Data.objects.create(
                symbol=row['symbol'],
                timestamp=row['timestamp'],
                open=row['open'],
                high=row['high'],
                low=row['low'],
                close=row['close'],
                volume=row['volume'],
                trade_count=row['trade_count'],
                vwap=row['vwap'],
                ema=row['ema'],
                label=row['label']
            )
        return JsonResponse({'message': 'Data successfully uploaded'}, status=201)

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
    

@csrf_exempt
@api_view(['POST'])
@parser_classes([MultiPartParser])
def upload_rsi(request):
    if 'file' not in request.FILES:
        return JsonResponse({'error': 'No file uploaded'}, status=400)

    try:
        # Read the uploaded file
        csv_file = request.FILES['file']
        df = pd.read_csv(csv_file)

        # Insert each row into the database
        for _, row in df.iterrows():
            RSI_Data.objects.create(
                symbol=row['symbol'],
                timestamp=row['timestamp'],
                open=row['open'],
                high=row['high'],
                low=row['low'],
                close=row['close'],
                volume=row['volume'],
                trade_count=row['trade_count'],
                vwap=row['vwap'],
                rsi=row['rsi'],
                label=row['label']
            )
        return JsonResponse({'message': 'Data successfully uploaded'}, status=201)

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

# GET endpoint to get the processed data for the ML model for EMA
@api_view(['GET'])
def get_ema_data(request):
    try:
        # Get data from SQLite DB
        macd_data = EMA_Data.objects.all()
        # Convert results above to a list of dictionaries
        data = list(macd_data.values(
            'symbol', 'timestamp', 'open', 'high', 'low', 
            'close', 'volume', 'vwap', 'trade_count',
            'ema', 'label'
        ))

        df = pd.DataFrame(data)

        # create CSV response
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename="ema_data.csv"'

        # Write dataframe to the response as CSV
        df.to_csv(path_or_buf=response, index=False)

        return response

    except Exception as e:
        # handle exceptions and return an error response
        return HttpResponse(f"Error: {str(e)}", content_type="text/plain", status=500)


# GET endpoint to get the processed data for the ML model for MACD
@api_view(['GET'])
def get_macd_data(request):
    try:
        # Get data from SQLite DB
        macd_data = MACD_Data.objects.all()
        # Convert results above to a list of dictionaries
        data = list(macd_data.values(
            'symbol', 'timestamp', 'open', 'high', 'low', 
            'close', 'volume', 'trade_count', 'vwap', 
            'macd', 'signal_line', 'label'
        ))

        df = pd.DataFrame(data)

        # create CSV response
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename="macd_data.csv"'

        # Write dataframe to the response as CSV
        df.to_csv(path_or_buf=response, index=False)

        return response

    except Exception as e:
        # handle exceptions and return an error response
        return HttpResponse(f"Error: {str(e)}", content_type="text/plain", status=500)
    


    # GET endpoint to get the processed data for the ML model for RSI
@api_view(['GET'])
def get_rsi_data(request):
    try:
        # Get data from SQLite DB
        rsi_Data = RSI_Data.objects.all()
        # Convert results above to a list of dictionaries
        data = list(rsi_Data.values(
            'timestamp', 'symbol', 'open', 'high', 'low', 
            'close', 'volume', 'vwap', 'trade_count',
            'rsi', 'label'
        ))

        df = pd.DataFrame(data)

        # create CSV response
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename="rsi_data.csv"'

        # Write dataframe to the response as CSV
        df.to_csv(path_or_buf=response, index=False)

        return response

    except Exception as e:
        # handle exceptions and return an error response
        return HttpResponse(f"Error: {str(e)}", content_type="text/plain", status=500)

@csrf_exempt
@api_view(['POST'])
def upload_metadata(request):
    if 'file' not in request.FILES:
        return JsonResponse({'error': 'No file uploaded'}, status=400)

    uploaded_file = request.FILES['file']

    try:
        # Read the file content and parse it as JSON
        metadata = json.load(uploaded_file)

        # Get the original file name
        original_filename = uploaded_file.name

        # Ensure the metadata directory exists
        metadata_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../metadata")
        os.makedirs(metadata_dir, exist_ok=True)

        # Construct the full path using the original file name
        metadata_file_path = os.path.join(metadata_dir, original_filename)

        # Save the metadata to the specified file
        with open(metadata_file_path, "w") as metadata_file:
            json.dump(metadata, metadata_file, indent=4)

        return JsonResponse({'message': 'File uploaded successfully', 'metadata': metadata}, status=201)
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON file'}, status=400)


# @csrf_exempt
# @api_view(['POST'])
# @parser_classes([MultiPartParser])
# def upload_model(request):
#     if 'file' not in request.FILES:
#         return JsonResponse({'error': 'No file uploaded'}, status=400)

#     try:
#         pkl_file = request.FILES['file']
#         file_name = pkl_file.name

#         # path to save the file
#         save_path = os.path.join(settings.BASE_DIR, 'models', file_name)
#         # make the directory if doesnt exist
#         os.makedirs(os.path.dirname(save_path), exist_ok=True)
#         #save
#         with open(save_path, 'wb') as f:
#             for chunk in pkl_file.chunks():
#                 f.write(chunk)

#         return JsonResponse({'message': 'File successfully uploaded'}, status=201)

#     except Exception as e:
#         return JsonResponse({'error': str(e)}, status=500)
    

@csrf_exempt
@api_view(['POST'])
@parser_classes([MultiPartParser])
def upload_model(request):
    if "file" not in request.FILES:
        return JsonResponse({"error": "No file uploaded"}, status=400)

    model_file = request.FILES["file"]
    strategy = request.POST.get("strategy")
    timestamp = request.POST.get("timestamp")

    if not strategy:
        return JsonResponse({"error": "Strategy is required."}, status=400)
    if not timestamp:
        return JsonResponse({"error": "Timestamp is required."}, status=400)

    # Save the file to strategy-specific directory
    strategy_dir = os.path.join(settings.BASE_DIR, f"stock_app/inference/models/{strategy}/")
    os.makedirs(strategy_dir, exist_ok=True)

    model_path = os.path.join(strategy_dir, f"{timestamp}_{model_file.name}")
    try:
        with open(model_path, "wb") as f:
            for chunk in model_file.chunks():
                f.write(chunk)
        return JsonResponse({"message": "Model uploaded successfully."}, status=201)
    except Exception as e:
        return JsonResponse({"error": f"Failed to save model: {str(e)}"}, status=500)

@api_view(['GET'])
def make_prediction(request, strategy, stock_symbol):
    try:
        strategy = strategy.lower()
        print(f"make_prediction called with strategy={strategy} and stock_symbol={stock_symbol}")
        
        # Validate inputs
        if not stock_symbol or not strategy:
            return JsonResponse({"error": "Missing stock_symbol or strategy."}, status=400)
        
        # Define the mapping of user-friendly stock names to Alpaca-compatible symbols
        stock_symbol_mapping = {
            "Nvidia": "NVDA",
            "Apple": "AAPL",
            "Microsoft": "MSFT",
            "Amazon": "AMZN",
            "Google": "GOOGL",
            "Meta": "META",
            "Tesla": "TSLA",
            "Berkshire Hathaway": "BRK.B",
            "Taiwan Semiconductors": "TSM",
            "Broadcom": "AVGO"
        }

        # Map the user-selected stock name to the Alpaca-compatible symbol
        alpaca_symbol = stock_symbol_mapping.get(stock_symbol)
        if not alpaca_symbol:
            return JsonResponse({"error": f"Invalid stock symbol: {stock_symbol}"}, status=400)

        # Fetch stock data
        print("Fetching stock data...")
        yesterday = (datetime.now() - timedelta(days=1)).date()
        start_date = yesterday - timedelta(days=15)  # Fetch at least 15 days of data for EMA
        stock_data = fetch_stock_data(alpaca_symbol, start_date=start_date, end_date=yesterday)
        
        if stock_data.empty:
            raise ValueError(f"No data fetched for {alpaca_symbol} from {start_date} to {yesterday}.")
        
        print(f"Fetched stock data:\n{stock_data.head()}")

        # Preprocess the stock data
        processed_data = preprocess_data(stock_data, strategy)
        print(f"Processed data:\n{processed_data}")

        # Run inference
        predicted_action = run_inference(processed_data, strategy)
        print(f"Predicted action: {predicted_action}")

        # Return the predictions as a response
        return JsonResponse({
            "stock_symbol": stock_symbol,
            "strategy": strategy,
            "prediction": predicted_action
        })

    except Exception as e:
        print(f"Error in make_prediction: {e}")
        return JsonResponse({"error": str(e)}, status=500)

